"""Census of row-pick sites: every place that picks rows by an order, so a new untied pick fails CI (PE-36).

Scope: `src/pitch_edge` (vendored code under `vendor/` excluded) and `research/studies/**/*.py`, test files
(`test_*.py`, `*_test.py`, `conftest.py`) excluded.

Detection, by pattern name:

- SQL, by regex over the text of every string literal that is not a docstring. An f-string is read as its
  literal parts with `{}` for each replacement field; adjacent literals are one string to the parser already.
  `sql:arg_max` / `sql:arg_min` (also `max_by` / `min_by`), `sql:first` / `sql:last` (`first(` / `last(`, no
  space before the parenthesis, so prose such as "run it first (...)" is not a site),
  `sql:any_value`, `sql:row_number` / `sql:rank` / `sql:dense_rank` (`...() OVER`), `sql:distinct_on`,
  `sql:asof` (`ASOF` join), and `sql:order_by_limit` (`ORDER BY` followed by `LIMIT` in the same string).
- pandas and builtins, from the AST:
  - `drop_duplicates`: a call with `subset` (positional or keyword); a full-row dedupe drops only identical rows.
  - `groupby.first` / `.last` / `.nth` / `.head` / `.tail`: called on a `groupby(...)` result, directly or through
    column selections (`g["x"]`, `g.x`), or on a name assigned such a result in the same function.
  - `agg.first` / `agg.last`: the string `"first"` / `"last"` passed to `agg`, `aggregate`, `transform` or
    `pivot_table` (as a value, in a tuple, list or dict, or as `aggfunc=`).
  - `idxmax` / `idxmin`.
  - `sort_values` (no stable kind) and `sort_values[stable]` (`kind="stable"` or `"mergesort"`): every call, so
    the allowlist states the total-order key the sort relies on, or why its order is not used for a pick.
  - `sort_values.pick`: `.iloc[0]`, `.iloc[-1]`, `.head(1)` or `.tail(1)` on a `sort_values(...)` result, column
    selections and `reset_index` in between allowed.
  - `max_key` / `min_key`: builtin `max` / `min` with `key=`.

Each site is keyed `<path from the repo root>::<enclosing function qualname, or <module>>::<pattern>`, with
`#<n>` (1-based, source order) appended when one function holds several sites of one pattern. Line numbers are
not part of the key. `tests/row_pick_allowlist.txt` lists every site as `<key> | <total-order key or reason>`;
the census fails on an unlisted site, on an entry with no matching site, and on an entry without a reason.
"""

from __future__ import annotations

import ast
import re
from collections import defaultdict
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

REPO = Path(__file__).resolve().parents[2]
ROOTS = ("src/pitch_edge", "research/studies")
ALLOWLIST = REPO / "tests" / "row_pick_allowlist.txt"
EXCLUDED_DIRS = frozenset({"vendor", "tests", "__pycache__"})

SQL_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("sql:arg_max", re.compile(r"\b(?:arg_max|max_by)\s*\(", re.I)),
    ("sql:arg_min", re.compile(r"\b(?:arg_min|min_by)\s*\(", re.I)),
    ("sql:first", re.compile(r"\bfirst\(", re.I)),
    ("sql:last", re.compile(r"\blast\(", re.I)),
    ("sql:any_value", re.compile(r"\bany_value\s*\(", re.I)),
    ("sql:row_number", re.compile(r"\brow_number\s*\(\s*\)\s*over\b", re.I)),
    ("sql:rank", re.compile(r"(?<!dense_)\brank\s*\(\s*\)\s*over\b", re.I)),
    ("sql:dense_rank", re.compile(r"\bdense_rank\s*\(\s*\)\s*over\b", re.I)),
    ("sql:distinct_on", re.compile(r"\bdistinct\s+on\b", re.I)),
    ("sql:asof", re.compile(r"\basof\b", re.I)),
    ("sql:order_by_limit", re.compile(r"\border\s+by\b.*?\blimit\b", re.I | re.S)),
)
GROUPBY_PICKS = frozenset({"first", "last", "nth", "head", "tail"})
AGG_CALLS = frozenset({"agg", "aggregate", "transform", "pivot_table"})
STABLE_KINDS = frozenset({"stable", "mergesort"})


def _is_test_file(path: Path) -> bool:
    name = path.name
    return name.startswith("test_") or name.endswith("_test.py") or name == "conftest.py"


def _docstring_nodes(tree: ast.AST) -> set[int]:
    """ids of the string constants that are docstrings of the module, a class or a function."""
    out: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef) and node.body:
            first = node.body[0]
            if (
                isinstance(first, ast.Expr)
                and isinstance(first.value, ast.Constant)
                and isinstance(first.value.value, str)
            ):
                out.add(id(first.value))
    return out


def _string_text(node: ast.AST) -> str | None:
    """The literal text of a string constant or f-string (`{}` per replacement field), else None."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        return "".join(
            v.value if isinstance(v, ast.Constant) and isinstance(v.value, str) else "{}" for v in node.values
        )
    return None


def _call_attr(node: ast.AST) -> str | None:
    """`name` for a call of the form `<expr>.name(...)`, else None."""
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
        return node.func.attr
    return None


def _strip_columns(node: ast.AST) -> ast.AST:
    """`node` without trailing column selections (`x["a"]`, `x[["a", "b"]]`, `x.a`)."""
    while isinstance(node, ast.Subscript | ast.Attribute):
        node = node.value
    return node


def _is_groupby(node: ast.AST, groupby_names: set[str]) -> bool:
    """True when `node`, past column selections, is a `groupby(...)` call or a name bound to one."""
    node = _strip_columns(node)
    return _call_attr(node) == "groupby" or (isinstance(node, ast.Name) and node.id in groupby_names)


def _agg_strings(node: ast.AST):
    """Every string constant in `node`, descending into tuples, lists and dicts."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        yield node.value
    elif isinstance(node, ast.Tuple | ast.List):
        for elt in node.elts:
            yield from _agg_strings(elt)
    elif isinstance(node, ast.Dict):
        for value in node.values:
            yield from _agg_strings(value)


def _is_sorted_result(node: ast.AST) -> bool:
    """True when `node` is a `sort_values(...)` call, possibly behind column selections and `reset_index`."""
    while True:
        if _call_attr(node) == "sort_values":
            return True
        if isinstance(node, ast.Subscript):
            node = node.value
        elif _call_attr(node) == "reset_index":
            node = node.func.value  # type: ignore[attr-defined]
        else:
            return False


def _int_value(node: ast.AST) -> int | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub) and isinstance(node.operand, ast.Constant):
        return -node.operand.value if isinstance(node.operand.value, int) else None
    return None


class _Scanner(ast.NodeVisitor):
    def __init__(self, docstrings: set[int]):
        self.docstrings = docstrings
        self.scope: list[str] = []
        self.groupby_names: list[set[str]] = [set()]
        self.sites: list[tuple[str, str, int]] = []  # (qualname, pattern, line)

    # ---------------------------------------------------------------- scopes
    def _qualname(self) -> str:
        return ".".join(self.scope) if self.scope else "<module>"

    def _enter(self, node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef) -> None:
        self.scope.append(node.name)
        self.groupby_names.append(set())
        self.generic_visit(node)
        self.groupby_names.pop()
        self.scope.pop()

    visit_FunctionDef = _enter
    visit_AsyncFunctionDef = _enter
    visit_ClassDef = _enter

    def _add(self, pattern: str, node: ast.AST) -> None:
        self.sites.append((self._qualname(), pattern, node.lineno))  # type: ignore[attr-defined]

    # ---------------------------------------------------------------- statements
    def visit_Assign(self, node: ast.Assign) -> None:
        if _is_groupby(node.value, self.groupby_names[-1]):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    self.groupby_names[-1].add(t.id)
        self.generic_visit(node)

    # ---------------------------------------------------------------- strings
    def _strings(self, node: ast.Constant | ast.JoinedStr) -> None:
        if id(node) in self.docstrings:
            return
        text = _string_text(node)
        if text:
            for pattern, rx in SQL_PATTERNS:
                for _ in rx.finditer(text):
                    self._add(pattern, node)

    def visit_Constant(self, node: ast.Constant) -> None:
        self._strings(node)

    def visit_JoinedStr(self, node: ast.JoinedStr) -> None:
        self._strings(node)  # the literal parts are read here; their nested constants are not visited again

    # ---------------------------------------------------------------- calls and subscripts
    def visit_Call(self, node: ast.Call) -> None:
        attr = _call_attr(node)
        kwargs = {k.arg: k.value for k in node.keywords if k.arg}
        if attr == "drop_duplicates" and (node.args or "subset" in kwargs):
            self._add("drop_duplicates", node)
        elif attr in GROUPBY_PICKS and _is_groupby(node.func.value, self.groupby_names[-1]):  # type: ignore[attr-defined]
            self._add(f"groupby.{attr}", node)
        elif attr in AGG_CALLS:
            for value in [*node.args, *kwargs.values()]:
                for text in _agg_strings(value):
                    if text in ("first", "last"):
                        self._add(f"agg.{text}", node)
        elif attr in ("idxmax", "idxmin"):
            self._add(attr, node)
        elif attr == "sort_values":
            kind = kwargs.get("kind")
            stable = isinstance(kind, ast.Constant) and kind.value in STABLE_KINDS
            self._add("sort_values[stable]" if stable else "sort_values", node)
        if attr in ("head", "tail") and node.args and _int_value(node.args[0]) == 1:
            if _is_sorted_result(node.func.value):  # type: ignore[attr-defined]
                self._add("sort_values.pick", node)
        if isinstance(node.func, ast.Name) and node.func.id in ("max", "min") and "key" in kwargs:
            self._add(f"{node.func.id}_key", node)
        self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript) -> None:
        if (
            isinstance(node.value, ast.Attribute)
            and node.value.attr == "iloc"
            and _int_value(node.slice) in (0, -1)
            and _is_sorted_result(node.value.value)
        ):
            self._add("sort_values.pick", node)
        self.generic_visit(node)


def row_picks(source: str) -> list[tuple[str, str, int]]:
    """(enclosing qualname, pattern, line) of every row-pick site in `source`, in source order."""
    tree = ast.parse(source)
    scanner = _Scanner(_docstring_nodes(tree))
    scanner.visit(tree)
    return sorted(scanner.sites, key=lambda s: s[2])


def scan(base: Path, roots: tuple[str, ...] = ROOTS) -> dict[str, int]:
    """`<path from base>::<qualname>::<pattern>[#n]` -> line for every site under `base / root`."""
    out: dict[str, int] = {}
    for root in roots:
        for path in sorted((base / root).rglob("*.py")):
            rel = path.relative_to(base)
            if EXCLUDED_DIRS & set(path.relative_to(base / root).parts[:-1]) or _is_test_file(path):
                continue
            groups: dict[tuple[str, str], list[int]] = defaultdict(list)
            for qual, pattern, line in row_picks(path.read_text()):
                groups[(qual, pattern)].append(line)
            for (qual, pattern), lines in groups.items():
                for i, line in enumerate(lines, 1):
                    suffix = f"#{i}" if len(lines) > 1 else ""
                    out[f"{rel.as_posix()}::{qual}::{pattern}{suffix}"] = line
    return out


def load_allowlist(path: Path) -> dict[str, str]:
    """`<key>` -> reason; blank lines and `#` comments are ignored."""
    out: dict[str, str] = {}
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        key, _, reason = (part.strip() for part in line.partition("|"))
        out[key] = reason
    return out


def census_problems(found: dict[str, int], allowed: dict[str, str]) -> list[str]:
    """Unlisted sites, stale entries and entries without a reason, one message each (empty when clean)."""
    problems = [f"unlisted row-pick site: {key} (line {found[key]})" for key in found if key not in allowed]
    problems += [f"stale allowlist entry (no such row-pick site): {key}" for key in allowed if key not in found]
    problems += [f"allowlist entry without a reason: {key}" for key, why in allowed.items() if not why]
    return problems


def test_every_row_pick_site_is_on_the_allowlist():
    problems = census_problems(scan(REPO), load_allowlist(ALLOWLIST))
    assert not problems, "\n".join(problems)


PLANTED = '''"""Docstrings that mention arg_max(x, t) or ORDER BY t LIMIT 1 are not sites."""
import pandas as pd


def latest(con, table):
    return con.execute(f"SELECT k, arg_max(v, t) FROM {table} GROUP BY k").df()


def dedupe(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop_duplicates("k").drop_duplicates()
'''


def test_a_planted_untied_pick_is_reported(tmp_path):
    pkg = tmp_path / "pkg"
    (pkg / "vendor").mkdir(parents=True)
    (pkg / "planted.py").write_text(PLANTED)
    (pkg / "test_skipped.py").write_text("def test_x(df):\n    return df.drop_duplicates('k')\n")
    (pkg / "vendor" / "skipped.py").write_text("def f(df):\n    return df.idxmax()\n")
    found = scan(tmp_path, ("pkg",))
    assert found == {"pkg/planted.py::latest::sql:arg_max": 6, "pkg/planted.py::dedupe::drop_duplicates": 10}
    assert census_problems(found, {}) == [
        "unlisted row-pick site: pkg/planted.py::latest::sql:arg_max (line 6)",
        "unlisted row-pick site: pkg/planted.py::dedupe::drop_duplicates (line 10)",
    ]


def test_stale_and_reasonless_allowlist_entries_are_reported():
    found = {"a.py::f::idxmax": 3}
    allowed = {"a.py::f::idxmax": "", "b.py::g::sql:any_value": "constant per group"}
    assert census_problems(found, allowed) == [
        "stale allowlist entry (no such row-pick site): b.py::g::sql:any_value",
        "allowlist entry without a reason: a.py::f::idxmax",
    ]


def test_site_keys_do_not_move_when_a_line_is_inserted_above(tmp_path):
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "m.py").write_text(PLANTED)
    before = scan(tmp_path, ("pkg",))
    (pkg / "m.py").write_text(PLANTED.replace("import pandas as pd\n", "import pandas as pd\nimport numpy as np\n\n"))
    after = scan(tmp_path, ("pkg",))
    assert set(after) == set(before)
    assert all(after[k] == before[k] + 2 for k in before)


def test_detection_rules_on_a_sampler():
    source = """
import pandas as pd

SQL = "SELECT * FROM t ORDER BY a DESC " "LIMIT 1"


class Store:
    def picks(self, df, g, xs):
        q = f"SELECT any_value(x), first(y), row_number() OVER (ORDER BY z) FROM {g} ASOF JOIN u"
        grouped = df.groupby("k")
        a = grouped["v"].last()
        b = df.groupby("k").head(2)
        c = df["v"].idxmax()
        d = df.sort_values("t")["v"].iloc[-1]
        e = df.sort_values(["t", "id"], kind="stable").reset_index(drop=True).head(1)
        f = max(xs, key=len)
        h = df.drop_duplicates()
        i = df.head(1)
        j = df.groupby("k").agg(["size", "sum"]).sort_values("size").head(3)
        k = df.groupby("k").agg(v=("v", "first")).pivot_table(index="k", values="v", aggfunc="last")
        msg = "run --select-only first (no selection.json)"
        return q, a, b, c, d, e, f, h, i, j, k, msg
"""
    sites = {(q, p) for q, p, _ in row_picks(source)}
    assert sites == {
        ("<module>", "sql:order_by_limit"),
        ("Store.picks", "sql:any_value"),
        ("Store.picks", "sql:first"),
        ("Store.picks", "sql:row_number"),
        ("Store.picks", "sql:asof"),
        ("Store.picks", "groupby.last"),
        ("Store.picks", "groupby.head"),
        ("Store.picks", "idxmax"),
        ("Store.picks", "sort_values"),
        ("Store.picks", "sort_values.pick"),
        ("Store.picks", "sort_values[stable]"),
        ("Store.picks", "max_key"),
        ("Store.picks", "agg.first"),
        ("Store.picks", "agg.last"),
    }
