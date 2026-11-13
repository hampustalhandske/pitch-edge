"""`pitch-edge research ...` — register hypotheses, import study-local trials, record reviews, list
hypotheses with their best rung, grade one on the evidence ladder, and print its ledger rows. Every
command takes `--ledger` (default `<data_dir>/research/ledger.duckdb`); `status`, `grade` and `ledger`
only read and never create the ledger file."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import typer
from pydantic import ValidationError
from rich.console import Console
from rich.table import Table

from pitch_edge.research.gates import grade as grade_hypothesis
from pitch_edge.research.importer import import_trials_csv
from pitch_edge.research.ledger import REVIEW_KINDS, Ledger, ProtocolError
from pitch_edge.research.registry import load

research_app = typer.Typer(help="Pre-registered hypotheses, the trials ledger and the L0-L5 evidence ladder.")

REGISTRATION_ARGUMENT = typer.Argument(..., help="research/registry/<id>.yaml")
CSV_ARGUMENT = typer.Argument(..., help="research/studies/<study>/trials.csv")
PRIMARY_OPTION = typer.Option([], "--primary", help="CSV metric name(s) that are the primary metric (repeatable)")
ALIAS_OPTION = typer.Option(
    [], "--alias", help="CSV_NAME=REGISTERED_PRIMARY_METRIC (exact string; repeatable), added to the built-in aliases"
)
LEDGER_OPTION = typer.Option(None, "--ledger", help="Ledger file (default: <data_dir>/research/ledger.duckdb)")
EXPORT_OUT_OPTION = typer.Option(..., "--out", help="JSONL file to write")


def _console() -> Console:
    # Piped output (and tests) get a wide fixed width so criterion names are never wrapped mid-word.
    return Console() if sys.stdout.isatty() else Console(width=220)


def _ledger(path: Path | None) -> Ledger:
    return Ledger(path)


def _print_table(df: pd.DataFrame, title: str) -> None:
    t = Table(title=title)
    for c in df.columns:
        t.add_column(str(c))
    for row in df.itertuples(index=False):
        t.add_row(*["" if v is None else (f"{v:.4g}" if isinstance(v, float) else str(v)) for v in row])
    _console().print(t)


@research_app.command("register")
def register(yaml_path: Path = REGISTRATION_ARGUMENT, ledger: Path | None = LEDGER_OPTION) -> None:
    """Record a registration file's sha256 in the ledger (refused if the id exists with another hash)."""
    console = _console()
    try:
        reg = load(yaml_path)
    except (ValidationError, ValueError, OSError) as exc:
        console.print(f"[red]invalid registration[/red] {yaml_path}: {exc}")
        raise typer.Exit(1) from exc
    try:
        new = _ledger(ledger).register(reg)
    except ProtocolError as exc:
        console.print(f"[red]refused:[/red] {exc}")
        raise typer.Exit(1) from exc
    state = "registered" if new else "already registered (unchanged)"
    console.print(
        f"{reg.id}: {state} — family {reg.family}, sha256 {reg.source_sha256}, {len(reg.expand_grid())} trial config(s)"
    )


@research_app.command("status")
def status(
    family: str | None = typer.Option(None, "--family", help="Only this multiple-testing family"),
    ledger: Path | None = LEDGER_OPTION,
) -> None:
    """One row per registered hypothesis: family, trial runs, distinct configs run, best rung."""
    led = _ledger(ledger)
    regs = led.registrations(family)
    rows = []
    for _, r in regs.iterrows():
        trials = led.trials(hypothesis_id=r["id"])
        rows.append(
            {
                "id": r["id"],
                "family": r["family"],
                "runs": len(trials),
                "configs": int(trials["config_hash"].nunique()) if not trials.empty else 0,
                "rung": grade_hypothesis(r["id"], led).rung,
                "registered": str(r["recorded_at"])[:19],
            }
        )
    _print_table(
        pd.DataFrame(rows, columns=["id", "family", "runs", "configs", "rung", "registered"]),
        f"research hypotheses{f' — {family}' if family else ''}",
    )


@research_app.command("grade")
def grade(
    hypothesis_id: str = typer.Argument(...),
    config_hash: str | None = typer.Option(None, "--config-hash", help="Grade this grid point (default: the best)"),
    ledger: Path | None = LEDGER_OPTION,
) -> None:
    """Print every L0-L5 criterion for one hypothesis with its status and the number behind it."""
    g = grade_hypothesis(hypothesis_id, _ledger(ledger), config_hash=config_hash)
    _print_table(g.table(), f"{hypothesis_id}: {g.rung}")
    if g.trial_id:
        _console().print(f"selected trial {g.trial_id} (config {g.config_hash})")


@research_app.command("ledger")
def ledger_rows(
    hypothesis_id: str = typer.Option(..., "--id", help="Hypothesis id"),
    ledger: Path | None = LEDGER_OPTION,
) -> None:
    """Print the hypothesis' trial rows (newest last) and reviews."""
    led = _ledger(ledger)
    trials = led.trials(hypothesis_id=hypothesis_id)
    cols = [
        "run_at",
        "trial_id",
        "config_json",
        "metric_name",
        "n",
        "metric_value",
        "ci_lo",
        "ci_hi",
        "p_value",
        "git_sha",
        "data_snapshot",
    ]
    view = trials[cols].assign(
        run_at=trials["run_at"].astype(str).str[:19],
        trial_id=trials["trial_id"].str[:12],
        git_sha=trials["git_sha"].str[:10],
    )
    _print_table(view, f"{hypothesis_id}: {len(trials)} trial run(s)")
    reviews = led.reviews(hypothesis_id)
    if not reviews.empty:
        reviews = reviews.assign(detail_json=reviews["detail_json"].map(lambda s: json.dumps(json.loads(s))))
        _print_table(reviews, f"{hypothesis_id}: reviews")


@research_app.command("export-ledger")
def export_ledger(
    family: str = typer.Option(..., "--family", help="Family whose trials the summary line counts"),
    upto: str = typer.Option(..., "--upto", help="row_hash of the last chain entry to export"),
    out: Path = EXPORT_OUT_OPTION,
    ledger: Path | None = LEDGER_OPTION,
) -> None:
    """Write the hash chain from genesis to --upto, the family's unchained rows and a trial-count summary as
    JSONL, for an independent recount of the family's tests (see `Ledger.export_chain`)."""
    led = _ledger(ledger)
    try:
        lines = led.export_chain(family, upto, led.metric_kinds(family))
    except ProtocolError as exc:
        _console().print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(line, sort_keys=True) + "\n" for line in lines))
    s = lines[-1]["summary"]
    n_chain = sum(1 for line in lines if "row_hash" in line)
    n_legacy = sum(1 for line in lines if line.get("chained") is False)
    _console().print(
        f"{out}: {n_chain} chain entries, {n_legacy} unchained {family} rows; {s['n_tests']} trials, "
        f"{s['n_distinct_hypothesis_config']} distinct (hypothesis, config)"
    )


@research_app.command("import-trials")
def import_trials(
    csv_path: Path = CSV_ARGUMENT,
    hypothesis_id: str = typer.Option(..., "--id", help="Registered hypothesis id whose rows to import"),
    primary: list[str] = PRIMARY_OPTION,
    alias: list[str] = ALIAS_OPTION,
    ledger: Path | None = LEDGER_OPTION,
) -> None:
    """Import a study-local trials.csv (R1 or R2 format) as `imported` trials; re-running adds nothing."""
    console = _console()
    try:
        aliases: dict[str, str] = {}
        for item in alias:
            name, sep, target = item.partition("=")
            if not (sep and name and target):
                raise ValueError("--alias takes CSV_NAME=REGISTERED_PRIMARY_METRIC")
            aliases[name] = target
        report = import_trials_csv(
            csv_path, hypothesis_id, _ledger(ledger), primary_metrics=primary or None, aliases=aliases or None
        )
    except (ValueError, OSError) as exc:
        console.print(f"[red]refused:[/red] {exc}")
        raise typer.Exit(1) from exc
    console.print(
        f"{hypothesis_id}: imported {report.imported}, {report.duplicates} already in the ledger, "
        f"{report.off_grid} off-grid, {report.other_ids} rows of other ids, {len(report.refused)} refused; "
        f"primary metric(s): {report.primary_metrics or 'none matched'}"
    )
    for reason in report.refused:
        console.print(f"  refused {reason}")


@research_app.command("review")
def review(
    hypothesis_id: str = typer.Argument(...),
    kind: str = typer.Option(..., "--kind", help=f"One of {', '.join(REVIEW_KINDS)}"),
    verdict: str = typer.Option(..., "--verdict", help="e.g. SIGNED / REJECTED, an edgelab grade, or 'settled'"),
    detail: str = typer.Option(
        "{}", "--detail", help='JSON object, e.g. {"file": "research/audits/x.md", "scope": {...}}'
    ),
    ledger: Path | None = LEDGER_OPTION,
) -> None:
    """Append a review (audit / edgelab / paper) or a pre-result registration amendment to the ledger."""
    console = _console()
    try:
        parsed = json.loads(detail)
        if not isinstance(parsed, dict):
            raise ValueError("--detail must be a JSON object")
        _ledger(ledger).record_review(hypothesis_id, kind, verdict, parsed)
    except (ValueError, ProtocolError) as exc:
        console.print(f"[red]refused:[/red] {exc}")
        raise typer.Exit(1) from exc
    console.print(f"{hypothesis_id}: recorded {kind} review {verdict!r}")
