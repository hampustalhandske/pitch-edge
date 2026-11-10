"""Census of module-level mutable state in `src/pitch_edge` (vendored code under `vendor/` excluded).

Every top-level assignment whose value is a mutable container (a dict, list or set literal or comprehension, or
a call to `dict`, `list`, `set`, `defaultdict`, `OrderedDict`, `deque`, `Counter` or `bytearray`) must be listed
in `tests/module_state_allowlist.txt` as `<path relative to src/pitch_edge>::<name> | <reason>`. "Top level"
means the module body, including the bodies of module-level `if`/`try`/`with` blocks, not functions or classes.
`__all__` is exempt (the import machinery's export list). The census fails on an unlisted object, on a listed
entry that no longer exists, and on an entry without a reason.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "src" / "pitch_edge"
ALLOWLIST = REPO / "tests" / "module_state_allowlist.txt"
EXCLUDED_DIRS = ("vendor",)
MUTABLE_CALLS = frozenset({"dict", "list", "set", "defaultdict", "OrderedDict", "deque", "Counter", "bytearray"})
EXEMPT_NAMES = frozenset({"__all__"})


def _is_mutable(value: ast.expr | None) -> bool:
    if isinstance(value, ast.Dict | ast.List | ast.Set | ast.DictComp | ast.ListComp | ast.SetComp):
        return True
    if isinstance(value, ast.Call):
        func = value.func
        name = func.id if isinstance(func, ast.Name) else func.attr if isinstance(func, ast.Attribute) else None
        return name in MUTABLE_CALLS
    return False


def _top_level(body: list[ast.stmt]):
    """Statements of a module body, descending into module-level if/try/with blocks."""
    for stmt in body:
        yield stmt
        if isinstance(stmt, ast.If):
            yield from _top_level(stmt.body)
            yield from _top_level(stmt.orelse)
        elif isinstance(stmt, ast.Try):
            yield from _top_level(stmt.body)
            for handler in stmt.handlers:
                yield from _top_level(handler.body)
            yield from _top_level(stmt.orelse)
            yield from _top_level(stmt.finalbody)
        elif isinstance(stmt, ast.With):
            yield from _top_level(stmt.body)


def mutable_globals(source: str) -> list[tuple[str, int]]:
    """(name, line) of every top-level assignment of a mutable container in `source`."""
    found: list[tuple[str, int]] = []
    for stmt in _top_level(ast.parse(source).body):
        if isinstance(stmt, ast.Assign):
            targets, value = stmt.targets, stmt.value
        elif isinstance(stmt, ast.AnnAssign) and stmt.value is not None:
            targets, value = [stmt.target], stmt.value
        else:
            continue
        for target in targets:
            pairs = (
                zip(target.elts, value.elts, strict=False)
                if isinstance(target, ast.Tuple) and isinstance(value, ast.Tuple)
                else [(target, value)]
            )
            for t, v in pairs:
                if isinstance(t, ast.Name) and t.id not in EXEMPT_NAMES and _is_mutable(v):
                    found.append((t.id, stmt.lineno))
    return found


def scan(root: Path) -> dict[str, int]:
    """`<path relative to root>::<name>` -> line for every module-level mutable container under `root`."""
    out: dict[str, int] = {}
    for path in sorted(root.rglob("*.py")):
        rel = path.relative_to(root)
        if rel.parts[0] in EXCLUDED_DIRS:
            continue
        for name, line in mutable_globals(path.read_text()):
            out[f"{rel.as_posix()}::{name}"] = line
    return out


def load_allowlist(path: Path) -> dict[str, str]:
    """`<path>::<name>` -> reason; blank lines and `#` comments are ignored."""
    out: dict[str, str] = {}
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        key, _, reason = (part.strip() for part in line.partition("|"))
        out[key] = reason
    return out


def census_problems(found: dict[str, int], allowed: dict[str, str]) -> list[str]:
    """Unlisted objects, stale entries and entries without a reason, one message each (empty when clean)."""
    problems = [
        f"unlisted module-level mutable state: {key} (line {found[key]})" for key in found if key not in allowed
    ]
    problems += [f"stale allowlist entry (no such module-level object): {key}" for key in allowed if key not in found]
    problems += [f"allowlist entry without a reason: {key}" for key, why in allowed.items() if not why]
    return problems


def test_every_module_level_mutable_object_is_on_the_allowlist():
    problems = census_problems(scan(SRC), load_allowlist(ALLOWLIST))
    assert not problems, "\n".join(problems)


def test_a_planted_module_level_dict_is_reported(tmp_path):
    pkg = tmp_path / "pkg"
    (pkg / "vendor").mkdir(parents=True)
    (pkg / "clean.py").write_text("X = (1, 2)\nY = frozenset({1})\n\ndef f():\n    z = {}\n    return z\n")
    (pkg / "planted.py").write_text(
        "import collections\n__all__ = ['a']\n_X = {}\nif True:\n    _Y: list[int] = []\n"
        "_Z = collections.defaultdict(list)\nA, B = 1, set()\n"
    )
    (pkg / "vendor" / "skipped.py").write_text("_V = {}\n")
    found = scan(pkg)
    assert found == {"planted.py::_X": 3, "planted.py::_Y": 5, "planted.py::_Z": 6, "planted.py::B": 7}
    assert census_problems(found, {}) == [
        "unlisted module-level mutable state: planted.py::_X (line 3)",
        "unlisted module-level mutable state: planted.py::_Y (line 5)",
        "unlisted module-level mutable state: planted.py::_Z (line 6)",
        "unlisted module-level mutable state: planted.py::B (line 7)",
    ]


def test_stale_and_reasonless_allowlist_entries_are_reported():
    found = {"a.py::_X": 1}
    allowed = {"a.py::_X": "", "b.py::_GONE": "constant lookup table, never mutated"}
    assert census_problems(found, allowed) == [
        "stale allowlist entry (no such module-level object): b.py::_GONE",
        "allowlist entry without a reason: a.py::_X",
    ]
