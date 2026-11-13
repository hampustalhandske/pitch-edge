"""Driver-side helpers of the determinism tests: canonical output text, state dumps and the planted kill.

A driver (`tests/determinism/drivers/*.py`) runs one pipeline in its own process and ends by writing a canonical
text file with `write_canonical`. Two runs of the same pipeline on the same inputs must produce byte-identical
files, so everything here is a total, process-independent function of the data:

* `canonical` turns frames, numpy and pandas scalars, timestamps and containers into plain JSON values. Floats keep
  their exact value (`json` writes `float.__repr__`, the shortest string that round-trips), NaN and infinities
  become the strings `"NaN"`, `"Infinity"`, `"-Infinity"`, and dict keys are sorted on output.
* `frame_records` turns a frame into its column list and its rows, sorted by each row's canonical JSON, so row
  order never matters but every value does.
* Columns named as volatile (ids drawn from `uuid4`, wall-clock stamps) are replaced by `VOLATILE` when set and
  stay None when null, so the comparison still sees whether they were filled. A volatile name that is not a column
  raises, so the named list cannot drift from the schema.
* `dump_duckdb` dumps every table of a DuckDB file that way: the state-equality comparison of engineering rule 2.

`kill_after` wraps a function so that the process SIGKILLs itself right after the function's n-th call returns,
the way a power cut or `kill -9` would end a run between two units of work.
"""

from __future__ import annotations

import datetime as dt
import json
import math
import os
import signal
import sys
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

VOLATILE = "<volatile>"
"""Stand-in for a value that differs between runs by design (a uuid4 id, a wall-clock stamp)."""


def canonical(value: Any) -> Any:
    """`value` as plain JSON data (see the module docstring). Raises TypeError for a type it does not know."""
    if value is None or isinstance(value, bool | str):
        return value
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, int | np.integer):
        return int(value)
    if isinstance(value, float | np.floating):
        f = float(value)
        if math.isnan(f):
            return "NaN"
        if math.isinf(f):
            return "Infinity" if f > 0 else "-Infinity"
        return f
    if value is pd.NaT:
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, dt.datetime | dt.date | dt.time):
        return value.isoformat()
    if isinstance(value, pd.Timedelta | dt.timedelta):
        return f"timedelta:{pd.Timedelta(value).value}ns"
    if isinstance(value, pd.DataFrame):
        return frame_records(value)
    if isinstance(value, pd.Series):
        return {"index": canonical(list(value.index)), "values": canonical(list(value.to_numpy()))}
    if isinstance(value, np.ndarray):
        return [canonical(v) for v in value.tolist()]
    if isinstance(value, Mapping):
        return {str(k): canonical(v) for k, v in value.items()}
    if isinstance(value, list | tuple | set | frozenset):
        items = [canonical(v) for v in value]
        return sorted(items, key=_sort_key) if isinstance(value, set | frozenset) else items
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, bytes):
        return value.hex()
    raise TypeError(f"no canonical form for {type(value).__name__}: {value!r}")


def _sort_key(value: Any) -> str:
    return json.dumps(value, sort_keys=True, allow_nan=False)


def _is_null(value: Any) -> bool:
    if value is None or value is pd.NaT:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def frame_records(
    df: pd.DataFrame,
    *,
    volatile: Iterable[str] = (),
    relabel: Mapping[str, Callable[[Any], Any]] | None = None,
) -> dict[str, Any]:
    """`df` as `{"columns": [[name, dtype], ...], "rows": [...]}`, rows sorted by their canonical JSON.

    `volatile` columns become `VOLATILE` (None stays None); `relabel` maps a column to a function applied to each of
    its values first (for example a uuid id replaced by a stable label). Raises KeyError for a named column that is
    not in `df`."""
    volatile = set(volatile)
    relabel = dict(relabel or {})
    missing = sorted((volatile | set(relabel)) - set(df.columns))
    if missing:
        raise KeyError(f"named columns {missing} are not in the frame (columns {list(df.columns)})")
    rows = []
    for rec in df.to_dict(orient="records"):
        out = {}
        for col, val in rec.items():
            if col in relabel:
                val = relabel[col](val)
            if col in volatile:
                val = None if _is_null(val) else VOLATILE
            out[str(col)] = canonical(val)
        rows.append(out)
    rows.sort(key=_sort_key)
    return {"columns": [[str(c), str(t)] for c, t in df.dtypes.items()], "rows": rows}


def dump_duckdb(
    path: str | Path,
    *,
    volatile: Mapping[str, Iterable[str]] | None = None,
    relabel: Mapping[str, Mapping[str, Callable[[Any], Any]]] | None = None,
) -> dict[str, Any]:
    """Every table of the DuckDB file at `path` (schema `main`), each as `frame_records`, keyed by table name.

    `volatile` and `relabel` are per table (see `frame_records`). Raises KeyError when they name a table the file
    does not hold, so a renamed table cannot silently fall out of the comparison."""
    import duckdb

    volatile = dict(volatile or {})
    relabel = dict(relabel or {})
    con = duckdb.connect(str(path), read_only=True)
    try:
        tables = [
            r[0]
            for r in con.execute(
                "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main' ORDER BY table_name"
            ).fetchall()
        ]
        missing = sorted((set(volatile) | set(relabel)) - set(tables))
        if missing:
            raise KeyError(f"named tables {missing} are not in {path} (tables {tables})")
        return {
            t: frame_records(
                con.execute(f'SELECT * FROM "{t}"').df(), volatile=volatile.get(t, ()), relabel=relabel.get(t)
            )
            for t in tables
        }
    finally:
        con.close()


def write_canonical(path: str | Path, obj: Any) -> None:
    """Write `obj` as canonical JSON text (sorted keys, one-space indent, no NaN literals, trailing newline),
    through a temporary file renamed into place so a killed run never leaves a partial output."""
    text = json.dumps(canonical(obj), sort_keys=True, indent=1, allow_nan=False, ensure_ascii=True) + "\n"
    target = Path(path)
    tmp = target.with_name(target.name + ".partial")
    tmp.write_text(text)
    tmp.replace(target)


def finish() -> None:
    """End the driver process with exit status 0 without interpreter teardown.

    Called after `write_canonical`, when the output is complete. On macOS, a process that loaded torch, DuckDB and
    pyarrow side by side can abort while their native libraries are torn down at exit ("recursive_mutex lock
    failed"), which would turn a finished run into a failed one."""
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


def kill_after(owner: Any, name: str, n: int, *, instead: Callable[..., Any] | None = None) -> None:
    """Replace `owner.<name>` by a wrapper that SIGKILLs this process right after the `n`-th call returns.

    `instead`, when given, is called in place of the original on the `n`-th call (with the same arguments) and the
    kill follows it: a planted defect such as a checkpoint written without its data. Calls before the `n`-th run the
    original unchanged."""
    original = getattr(owner, name)
    calls = {"n": 0}

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        calls["n"] += 1
        if calls["n"] < n:
            return original(*args, **kwargs)
        result = (instead or original)(*args, **kwargs)
        sys.stdout.flush()
        sys.stderr.flush()
        os.kill(os.getpid(), signal.SIGKILL)
        return result  # pragma: no cover - never reached

    setattr(owner, name, wrapper)


def driver_args(argv: list[str] | None = None, paths: Mapping[str, str] | None = None) -> Any:
    """The command line every driver takes: `--work DIR --out FILE [--defect NAME] [--kill-after N]
    [--kill-at NAME]`, plus one required path option per entry of `paths` (option name -> help text)."""
    import argparse

    p = argparse.ArgumentParser()
    for name, text in (paths or {}).items():
        p.add_argument(f"--{name}", required=True, type=Path, help=text)
    p.add_argument("--work", required=True, type=Path, help="working directory of this run (inputs and state)")
    p.add_argument("--out", required=True, type=Path, help="canonical output file to write")
    p.add_argument("--defect", default="", help="name of the planted defect to switch on (empty: none)")
    p.add_argument("--kill-after", default=0, type=int, help="SIGKILL after this many units of work (0: never)")
    p.add_argument("--kill-at", default="", help="which unit of work `--kill-after` counts")
    return p.parse_args(argv)
