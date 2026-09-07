"""The pre-match dossier: nine sections, every number traceable, gaps stated per fixture.

Design rule (the same "no unvalidated number reaches the UI" rule, applied to a new surface):

* A section never computes a new model number. Section 1 quotes a walk-forward prediction from
  `model_predictions` when the fixture was in a backtest fold, or a `fixture_predictions` row written
  by the CLI *before* the dossier is rendered (labelled "fitted at dossier time, raw"), together with
  the backtest evidence for that model (bits vs the closing price, per-league gap, empirical
  calibration band, model card).
* Each fact is rendered from a `values` dict that is *also* serialised as the source `Document`
  (`wh:<table>:<key>`), so `Dossier.verify()` — the RAG layer's `verify_citations` — can check
  mechanically that every figure in the text exists in a source. Templates contain no literal numbers.
* Section 9 is generated from a coverage checklist (confirmed lineup? referee assigned? early price?
  injury-report age? …) — the model's own epistemic humility, computed per fixture, not hand-written.
* Dossiers are versioned warehouse rows; re-running after new data lands gives a diff.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from pitch_edge.config import get_settings
from pitch_edge.data.storage import Warehouse
from pitch_edge.data.teams import normalise
from pitch_edge.intel.fixture import Fixture, find_fixture
from pitch_edge.rag.documents import Document
from pitch_edge.rag.generate import verify_citations

logger = logging.getLogger(__name__)

SECTIONS = [
    ("number", "THE NUMBER"),
    ("form", "FORM & STRENGTH"),
    ("lineup", "WHO'S ACTUALLY PLAYING"),
    ("referee", "THE REFEREE"),
    ("conditions", "CONDITIONS"),
    ("market", "MARKET BEHAVIOUR"),
    ("narrative", "NARRATIVE SIGNAL"),
    ("similar", "SIMILAR MATCHES"),
    ("gaps", "WHAT WOULD CHANGE THE ANSWER"),
]
OUTCOMES = ("home", "draw", "away")
_ID_SAFE = re.compile(r"[^A-Za-z0-9_:\-./]")


# ------------------------------------------------------------------ formatting
def pct(x) -> str:
    return "n/a" if x is None or pd.isna(x) else f"{float(x) * 100:.1f}%"


def spct(x) -> str:
    return "n/a" if x is None or pd.isna(x) else f"{float(x) * 100:+.1f}%"


def num(x, d: int = 2) -> str:
    return "n/a" if x is None or pd.isna(x) else f"{float(x):.{d}f}"


def snum(x, d: int = 2) -> str:
    return "n/a" if x is None or pd.isna(x) else f"{float(x):+.{d}f}"


def intg(x) -> str:
    return "n/a" if x is None or pd.isna(x) else f"{int(round(float(x)))}"


def day(x) -> str:
    return "n/a" if x is None or pd.isna(pd.Timestamp(x)) else pd.Timestamp(x).strftime("%Y-%m-%d")


def stamp(x) -> str:
    return "n/a" if x is None or pd.isna(pd.Timestamp(x)) else pd.Timestamp(x).strftime("%Y-%m-%d %H:%M")


def doc_id(*parts) -> str:
    return _ID_SAFE.sub("-", ":".join(str(p) for p in parts))


# ------------------------------------------------------------------ structures
@dataclass
class Claim:
    text: str
    sources: list[str]
    status: str = "ok"  # ok | provisional | missing

    def render(self) -> str:
        return f"{self.text} " + " ".join(f"[{s}]" for s in self.sources)


@dataclass
class Section:
    key: str
    title: str
    claims: list[Claim] = field(default_factory=list)

    @property
    def status(self) -> str:
        if not self.claims or all(c.status == "missing" for c in self.claims):
            return "missing"
        return "provisional" if any(c.status == "provisional" for c in self.claims) else "ok"


@dataclass
class Dossier:
    fixture: Fixture
    model_name: str
    generated_at: pd.Timestamp
    version: str
    sections: list[Section]
    documents: list[Document]
    coverage: dict
    market_series: list[dict] = field(default_factory=list)
    narrative: str | None = None
    narrative_backend: str | None = None

    @property
    def fixture_key(self) -> str:
        return self.fixture.fixture_key

    def section(self, key: str) -> Section:
        return next(s for s in self.sections if s.key == key)

    def header(self) -> str:
        f = self.fixture
        when = day(f.date) + (f", {f.kickoff_time} UTC" if f.kickoff_time else "")
        tag = {
            "warehouse_match": "played",
            "openfootball_upcoming": "scheduled",
            "hypothetical": "HYPOTHETICAL — not in any fixture list",
        }[f.source]
        return f"MATCH INTEL — {f.home_team} vs. {f.away_team} — {when} · {f.league_code or 'league n/a'} · {tag} · model {self.model_name} · v{self.version}"

    def markdown(self) -> str:
        bar = "═" * 80
        out = [self.header(), bar]
        for i, s in enumerate(self.sections, 1):
            flag = "" if s.status == "ok" else f"  ({s.status.upper()})"
            out.append(f"{i}. {s.title}{flag}")
            out += [f"   - {c.render()}" for c in s.claims] or ["   - (nothing available)"]
        if self.narrative:
            out += ["", f"Analyst narrative ({self.narrative_backend}, citation-verified):", self.narrative]
        out.append(bar)
        return "\n".join(out)

    def verify(self) -> tuple[bool, list[str]]:
        text = "\n".join(c.render() for s in self.sections for c in s.claims) + "\n" + self.header()
        ok, _cited, missing = verify_citations(text, self.documents)
        return ok, missing

    def to_json(self) -> str:
        return json.dumps(
            {
                "fixture": self.fixture.to_dict(),
                "model_name": self.model_name,
                "generated_at": self.generated_at.isoformat(timespec="seconds"),
                "version": self.version,
                "sections": [
                    {"key": s.key, "title": s.title, "status": s.status, "claims": [asdict(c) for c in s.claims]}
                    for s in self.sections
                ],
                "documents": [{"doc_id": d.doc_id, "text": d.text, "metadata": d.metadata} for d in self.documents],
                "coverage": self.coverage,
                "market_series": self.market_series,
                "narrative": self.narrative,
                "narrative_backend": self.narrative_backend,
            },
            default=str,
        )


# ------------------------------------------------------------------- builder
class DossierBuilder:
    def __init__(
        self,
        wh: Warehouse,
        model_name: str = "gbdt",
        reports_dir: str | Path | None = None,
        now: datetime | None = None,
        generator=None,
        fetch: bool = True,
    ):
        self.wh = wh
        self.model_name = model_name
        self.reports_dir = Path(reports_dir) if reports_dir else get_settings().reports_dir
        self.now = (
            pd.Timestamp(now or datetime.now(UTC)).tz_localize(None)
            if (now is None or now.tzinfo)
            else pd.Timestamp(now)
        )
        self.generator = generator
        self.fetch = fetch
        self._docs: dict[str, Document] = {}
        self.coverage: dict = {}
        self.market_series: list[dict] = []

    # ------------------------------------------------------------ plumbing
    def _fact(
        self,
        table: str,
        key: str,
        values: dict,
        template: str,
        status: str = "ok",
        extra_sources: list[str] | None = None,
        meta: dict | None = None,
    ) -> Claim:
        """Register a source document built from `values` and return the claim rendered from the same values."""
        did = doc_id("wh", table, key)
        vals = {k: ("n/a" if v is None else str(v)) for k, v in values.items()}
        text = f"{table}: " + "; ".join(f"{k}={v}" for k, v in vals.items())
        self._docs[did] = Document(did, text, {"type": "intel", "table": table, **(meta or {})})
        return Claim(template.format(**vals), [did, *(extra_sources or [])], status)

    def _missing(self, text: str, table: str, key: str, values: dict | None = None) -> Claim:
        return self._fact(table, key, values or {"status": "missing"}, text, status="missing")

    def _q(self, sql: str, params: list | None = None) -> pd.DataFrame:
        try:
            return self.wh.query(sql, params)
        except Exception as exc:  # noqa: BLE001 - a missing table is a gap, not a crash
            logger.debug("intel query failed (%s): %s", exc, sql[:80])
            return pd.DataFrame()

    def _has(self, table: str) -> bool:
        return self.wh.table_exists(table)

    # --------------------------------------------------------------- build
    def build(
        self, home: str, away: str, date=None, prediction: dict | None = None, fixture: Fixture | None = None
    ) -> Dossier:
        fx = fixture or find_fixture(self.wh, home, away, date, fetch_upcoming=self.fetch, now=self.now)
        self._docs, self.coverage, self.market_series = {}, {}, []
        as_of = min(self.now, pd.Timestamp(fx.date)) if fx.played and fx.date is not None else self.now
        ctx = {"fx": fx, "as_of": as_of}
        sections = [
            Section("number", "THE NUMBER", self._sec_number(ctx, prediction)),
            Section("form", "FORM & STRENGTH", self._sec_form(ctx)),
            Section("lineup", "WHO'S ACTUALLY PLAYING", self._sec_lineup(ctx)),
            Section("referee", "THE REFEREE", self._sec_referee(ctx)),
            Section("conditions", "CONDITIONS", self._sec_conditions(ctx)),
            Section("market", "MARKET BEHAVIOUR", self._sec_market(ctx)),
            Section("narrative", "NARRATIVE SIGNAL", self._sec_narrative(ctx)),
            Section("similar", "SIMILAR MATCHES", self._sec_similar(ctx)),
        ]
        sections.append(Section("gaps", "WHAT WOULD CHANGE THE ANSWER", self._sec_gaps(ctx)))
        version = self.now.strftime("%Y%m%dT%H%M%S")
        # header numbers (date, kickoff) must be grounded too
        self._fact(
            "intel_fixture",
            fx.fixture_key,
            {
                "home": fx.home_team,
                "away": fx.away_team,
                "date": day(fx.date),
                "kickoff": fx.kickoff_time or "n/a",
                "league": fx.league_code or "n/a",
                "source": fx.source,
                "version": version,
            },
            "{home} vs {away}",
        )
        dossier = Dossier(
            fx,
            self.model_name,
            self.now,
            version,
            sections,
            list(self._docs.values()),
            self.coverage,
            self.market_series,
        )
        if self.generator is not None and getattr(self.generator, "backend", "template") == "claude":
            self._narrate(dossier)
        return dossier

    def _narrate(self, dossier: Dossier) -> None:
        q = (
            f"Write a five-bullet pre-match analyst summary for {dossier.fixture.home_team} vs {dossier.fixture.away_team} "
            "using only the documents: the model view and how much to trust it, form, availability, market, and what is missing. "
            "Quote numbers exactly and cite every sentence."
        )
        try:
            ans = self.generator.answer(q, dossier.documents)
        except Exception as exc:  # noqa: BLE001
            logger.warning("narrative generation failed: %s", exc)
            return
        if ans.verified:
            dossier.narrative, dossier.narrative_backend = ans.text, ans.backend
        else:
            logger.warning("narrative dropped: unverified figures %s", ans.unverified_numbers)

    # ----------------------------------------------------------- section 1
    def _latest_run(self) -> str | None:
        rid = get_settings().artifacts_dir / "run_id.txt"
        if rid.exists():
            return rid.read_text().strip()
        df = (
            self._q("SELECT max(run_id) AS r FROM model_predictions")
            if self._has("model_predictions")
            else pd.DataFrame()
        )
        return None if df.empty or pd.isna(df["r"].iloc[0]) else str(df["r"].iloc[0])

    def _sec_number(self, ctx: dict, prediction: dict | None) -> list[Claim]:
        fx: Fixture = ctx["fx"]
        claims: list[Claim] = []
        run = self._latest_run()
        pred = pd.DataFrame()
        if fx.match_id and self._has("model_predictions") and run:
            pred = self._q(
                "SELECT * FROM model_predictions WHERE match_id = ? AND model_name = ? AND run_id = ?",
                [fx.match_id, self.model_name, run],
            )
        p_home = None
        if not pred.empty:
            r = pred.iloc[0]
            best = max(
                OUTCOMES, key=lambda o: float(r.get(f"edge_{o}", np.nan)) if pd.notna(r.get(f"edge_{o}")) else -9
            )
            vals = {
                "model": self.model_name,
                "run_id": run,
                "p_home": pct(r["p_home"]),
                "p_draw": pct(r["p_draw"]),
                "p_away": pct(r["p_away"]),
                "mkt_home": pct(r.get("mkt_home")),
                "mkt_draw": pct(r.get("mkt_draw")),
                "mkt_away": pct(r.get("mkt_away")),
                "best_side": best,
                "best_edge": spct(r.get(f"edge_{best}")),
                "fold": "walk-forward out-of-sample fold, isotonic-calibrated on past folds",
            }
            claims.append(
                self._fact(
                    "model_predictions",
                    f"{fx.match_id}:{self.model_name}:{run}",
                    vals,
                    "Model {model} ({fold}, run {run_id}): P(home) {p_home} · P(draw) {p_draw} · P(away) {p_away}; "
                    "no-vig market at the bet price {mkt_home} / {mkt_draw} / {mkt_away}; largest edge {best_side} {best_edge}.",
                )
            )
            p_home = float(r["p_home"])
            self.coverage["model_prediction_source"] = "backtest_fold"
            self.coverage["calibrated"] = True
        elif prediction is not None:
            vals = {
                "model": self.model_name,
                "version": prediction.get("version", "n/a"),
                "features_to": day(prediction.get("features_to")),
                "p_home": pct(prediction["p_home"]),
                "p_draw": pct(prediction["p_draw"]),
                "p_away": pct(prediction["p_away"]),
                "mkt_home": pct(prediction.get("mkt_home")),
                "mkt_draw": pct(prediction.get("mkt_draw")),
                "mkt_away": pct(prediction.get("mkt_away")),
                "market_source": prediction.get("market_source") or "no market quote for this fixture in the warehouse",
                "kind": "fitted at dossier time on the feature store, RAW (no fold calibration)",
            }
            claims.append(
                self._fact(
                    "fixture_predictions",
                    f"{fx.fixture_key}:{self.model_name}:{vals['version']}",
                    vals,
                    "Model {model} ({kind}; features to {features_to}, version {version}): P(home) {p_home} · P(draw) {p_draw} · P(away) {p_away}; "
                    "market reference {mkt_home} / {mkt_draw} / {mkt_away} ({market_source}).",
                    status="provisional",
                )
            )
            p_home = float(prediction["p_home"])
            self.coverage["model_prediction_source"] = "fitted_at_dossier_time_raw"
            self.coverage["calibrated"] = False
        else:
            claims.append(
                self._missing(
                    "No prediction for this fixture: it is outside the latest backtest run and no fixture prediction was computed.",
                    "model_predictions",
                    f"{fx.fixture_key}:none",
                )
            )
            self.coverage["model_prediction_source"] = None
            self.coverage["calibrated"] = False
        # backtest evidence
        if self._has("backtest_summaries") and run:
            bs = self._q(
                "SELECT * FROM backtest_summaries WHERE model = ? AND run_id = ? AND label = 'main' AND strategy = 'kelly_quarter' LIMIT 1",
                [self.model_name, run],
            )
            if not bs.empty:
                b = bs.iloc[0]
                ll, mll = (
                    float(b.get("multiclass_log_loss", np.nan)),
                    float(b.get("market_multiclass_log_loss", np.nan)),
                )
                bits = (mll - ll) / np.log(2) if pd.notna(ll) and pd.notna(mll) else np.nan
                vals = {
                    "model": self.model_name,
                    "run_id": run,
                    "log_loss": num(ll, 4),
                    "market_log_loss": num(mll, 4),
                    "bits": snum(bits, 3),
                    "n": intg(b.get("n_predictions")),
                    "roi": pct(b.get("roi")) if "roi" in b else "n/a",
                    "bet_price": str(b.get("bet_price_source")),
                    "closing_price": str(b.get("closing_price_source")),
                }
                claims.append(
                    self._fact(
                        "backtest_summaries",
                        f"main:{self.model_name}:{run}",
                        vals,
                        "Backtest evidence ({run_id}, main slice): {model} log-loss {log_loss} vs market {market_log_loss} = {bits} bits/match over {n} out-of-sample predictions; "
                        "a negative number means the closing price is more informative than this model. Bets priced at {bet_price}, CLV vs {closing_price}.",
                    )
                )
        by_league = self.reports_dir / "main" / "by_league.csv"
        if by_league.exists() and fx.league_code:
            bl = pd.read_csv(by_league)
            col = next((c for c in bl.columns if c.startswith("league")), None)
            row = bl[bl[col] == fx.league_code] if col else pd.DataFrame()
            if not row.empty:
                r = row.iloc[0]
                bits_col = next((c for c in bl.columns if "bits" in c), None)
                n_col = next((c for c in bl.columns if c in ("n", "n_predictions", "matches")), None)
                vals = {
                    "league": fx.league_code,
                    "bits": snum(r[bits_col], 3) if bits_col else "n/a",
                    "n": intg(r[n_col]) if n_col else "n/a",
                }
                claims.append(
                    self._fact(
                        "by_league_csv",
                        fx.league_code,
                        vals,
                        "In {league} specifically the market-aware GBDT sits {bits} bits from the closing price over {n} matches (reports/main/by_league.csv).",
                    )
                )
        # empirical calibration band around the quoted home probability
        if p_home is not None and self._has("model_predictions") and run:
            lo, hi = max(0.0, p_home - 0.05), min(1.0, p_home + 0.05)
            band = self._q(
                "SELECT count(*) AS n, avg(CASE WHEN result = 0 THEN 1 ELSE 0 END) AS obs, avg(p_home) AS mp FROM model_predictions WHERE model_name = ? AND run_id = ? AND p_home BETWEEN ? AND ?",
                [self.model_name, run, lo, hi],
            )
            if not band.empty and int(band["n"].iloc[0]) > 0:
                b = band.iloc[0]
                vals = {
                    "lo": pct(lo),
                    "hi": pct(hi),
                    "n": intg(b["n"]),
                    "observed": pct(b["obs"]),
                    "mean_pred": pct(b["mp"]),
                    "model": self.model_name,
                }
                claims.append(
                    self._fact(
                        "model_predictions_band",
                        f"{self.model_name}:{run}:{vals['lo']}",
                        vals,
                        "Confidence band: when {model} gave the home side {lo}–{hi} in the backtest, home actually won {observed} of {n} matches (mean prediction {mean_pred}).",
                    )
                )
        card = self.reports_dir / "main" / f"model_card_{self.model_name}.json"
        if card.exists():
            c = json.loads(card.read_text())
            vals = {
                "model": self.model_name,
                "n_features": intg(len(c.get("features", []))),
                "backend": str(c.get("backend", c.get("family", "n/a"))),
                "leakage": "; ".join(c.get("leakage_checks", [])) or "n/a",
            }
            claims.append(
                self._fact(
                    "model_card",
                    self.model_name,
                    vals,
                    "Model card {model}: {n_features} features, backend {backend}; leakage checks: {leakage}.",
                )
            )
        return claims

    # ----------------------------------------------------------- section 2
    def _team_rows(self, team: str, before: pd.Timestamp, n: int = 10) -> pd.DataFrame:
        if not self._has("features"):
            return pd.DataFrame()
        return self._q(
            "SELECT * FROM features WHERE (home_team = ? OR away_team = ?) AND date < ? ORDER BY date DESC LIMIT ?",
            [team, team, before, n],
        )

    def _sec_form(self, ctx: dict) -> list[Claim]:
        fx: Fixture = ctx["fx"]
        before = pd.Timestamp(fx.date) + pd.Timedelta(days=1) if fx.played else self.now + pd.Timedelta(days=1)
        if fx.played and fx.match_id:
            before = pd.Timestamp(fx.date)  # strictly pre-match
        claims: list[Claim] = []
        for team in (fx.home_team, fx.away_team):
            rows = self._team_rows(team, before, 10)
            if rows.empty:
                claims.append(
                    self._missing(
                        f"No feature-store history for {team} before {day(before)}.",
                        "features",
                        f"{normalise(team)}:none",
                    )
                )
                continue
            latest = rows.iloc[0]
            is_home = latest["home_team"] == team
            pre = "h_" if is_home else "a_"
            elo_col = "elo_home" if is_home else "elo_away"
            oldest = rows.iloc[-1]
            elo_then = oldest["elo_home"] if oldest["home_team"] == team else oldest["elo_away"]
            vals = {
                "team": team,
                "n": intg(len(rows)),
                "elo_now": num(latest[elo_col], 0),
                "elo_then": num(elo_then, 0),
                "elo_delta": snum(float(latest[elo_col]) - float(elo_then), 0),
                "ppg5": num(latest.get(f"{pre}pts_r5"), 2),
                "gf5": num(latest.get(f"{pre}gf_r5"), 2),
                "ga5": num(latest.get(f"{pre}ga_r5"), 2),
                "ppg10": num(latest.get(f"{pre}pts_r10"), 2),
                "as_of": day(latest["date"]),
                "w5": "5",
                "w10": "10",
            }
            claims.append(
                self._fact(
                    "features",
                    f"{normalise(team)}:{vals['as_of']}",
                    vals,
                    "{team}: Elo {elo_now} ({elo_delta} over the last {n} matches, from {elo_then}); last-{w5} form {ppg5} pts/game, {gf5} scored / {ga5} conceded per game; last-{w10} {ppg10} pts/game (feature store as of {as_of}).",
                )
            )
        if self._has("team_strength"):
            ts = self._q("SELECT * FROM team_strength WHERE team IN (?, ?)", [fx.home_team, fx.away_team])
            if not ts.empty:
                h = ts[ts["team"] == fx.home_team].head(1)
                a = ts[ts["team"] == fx.away_team].head(1)
                if not h.empty and not a.empty:
                    vals = {
                        "home": fx.home_team,
                        "away": fx.away_team,
                        "h_att": snum(h["attack"].iloc[0]),
                        "h_def": snum(h["defence"].iloc[0]),
                        "a_att": snum(a["attack"].iloc[0]),
                        "a_def": snum(a["defence"].iloc[0]),
                        "hfa": num(h["home_advantage"].iloc[0], 3),
                        "as_of": day(h["as_of"].iloc[0]),
                    }
                    claims.append(
                        self._fact(
                            "team_strength",
                            f"{fx.fixture_key}",
                            vals,
                            "Dixon-Coles (as of {as_of}): {home} attack {h_att} / defence {h_def}; {away} attack {a_att} / defence {a_def}; league home advantage {hfa}.",
                        )
                    )
        h2h = self._q(
            "SELECT date, home_team, away_team, home_goals, away_goals FROM matches WHERE ((home_team = ? AND away_team = ?) OR (home_team = ? AND away_team = ?)) AND date < ? ORDER BY date DESC LIMIT 5",
            [fx.home_team, fx.away_team, fx.away_team, fx.home_team, before],
        )
        if not h2h.empty:
            hw = int(
                (
                    (h2h["home_team"] == fx.home_team) & (h2h["home_goals"] > h2h["away_goals"])
                    | (h2h["away_team"] == fx.home_team) & (h2h["away_goals"] > h2h["home_goals"])
                ).sum()
            )
            dr = int((h2h["home_goals"] == h2h["away_goals"]).sum())
            vals = {
                "home": fx.home_team,
                "away": fx.away_team,
                "n": intg(len(h2h)),
                "home_wins": intg(hw),
                "draws": intg(dr),
                "away_wins": intg(len(h2h) - hw - dr),
                "list": "; ".join(
                    f"{day(r.date)} {r.home_team} {intg(r.home_goals)}-{intg(r.away_goals)} {r.away_team}"
                    for r in h2h.itertuples()
                ),
            }
            claims.append(
                self._fact(
                    "matches_h2h",
                    fx.fixture_key,
                    vals,
                    "Head-to-head, last {n}: {home} {home_wins} wins, {draws} draws, {away} {away_wins} wins ({list}).",
                )
            )
        split = []
        for team, col in ((fx.home_team, "home_team"), (fx.away_team, "away_team")):
            g = self._q(
                f"SELECT {col} AS t, home_goals, away_goals FROM matches WHERE {col} = ? AND date < ? ORDER BY date DESC LIMIT 19",
                [team, before],
            )
            if g.empty:
                continue
            gf, ga = (g["home_goals"], g["away_goals"]) if col == "home_team" else (g["away_goals"], g["home_goals"])
            pts = np.where(gf > ga, 3, np.where(gf == ga, 1, 0))
            split.append((team, "home" if col == "home_team" else "away", len(g), float(pts.mean())))
        if split:
            vals = {}
            for i, (team, venue, n, ppg) in enumerate(split):
                vals.update({f"team{i}": team, f"venue{i}": venue, f"n{i}": intg(n), f"ppg{i}": num(ppg, 2)})
            tmpl = " · ".join(
                f"{{team{i}}} at {{venue{i}}}: {{ppg{i}}} pts/game over the last {{n{i}}}" for i in range(len(split))
            )
            claims.append(self._fact("matches_split", fx.fixture_key, vals, "Home/away split — " + tmpl + "."))
        return claims

    # ----------------------------------------------------------- section 3
    def _sec_lineup(self, ctx: dict) -> list[Claim]:
        fx: Fixture = ctx["fx"]
        claims: list[Claim] = []
        confirmed = pd.DataFrame()
        if self._has("api_football_lineups"):
            cols = self.wh.columns("api_football_lineups")
            where = "team IN (?, ?)"
            params: list = [fx.home_team, fx.away_team]
            if "match_date" in cols and fx.date is not None:
                where += " AND CAST(match_date AS DATE) = ?"
                params.append(pd.Timestamp(fx.date).date())
            confirmed = self._q(f"SELECT * FROM api_football_lineups WHERE {where}", params)
        self.coverage["confirmed_lineup"] = not confirmed.empty
        if not confirmed.empty:
            self.coverage["lineup_source"] = "api_football"
            for team in (fx.home_team, fx.away_team):
                t = confirmed[confirmed["team"] == team]
                xi = t[t["slot"] == "start"]["player"].tolist() if "slot" in t else t["player"].tolist()
                if not xi:
                    continue
                vals = {
                    "team": team,
                    "formation": str(t["formation"].iloc[0]) if "formation" in t else "n/a",
                    "n": intg(len(xi)),
                    "xi": ", ".join(map(str, xi)),
                    "source": "API-Football confirmed lineup",
                }
                claims.append(
                    self._fact(
                        "api_football_lineups",
                        f"{fx.fixture_key}:{normalise(team)}",
                        vals,
                        "CONFIRMED XI {team} ({formation}, {n} starters, {source}): {xi}.",
                    )
                )
        else:
            self.coverage["lineup_source"] = "transfermarkt_last_xi" if self._has("tm_game_lineups") else None
            if self._has("tm_game_lineups") and self._has("tm_games"):
                from pitch_edge.features.context import club_id_map

                cmap = club_id_map(self.wh, [fx.home_team, fx.away_team])
                cutoff = pd.Timestamp(fx.date) if fx.date is not None else self.now
                for team in (fx.home_team, fx.away_team):
                    cid = cmap.get(team)
                    if cid is None:
                        claims.append(
                            self._missing(
                                f"{team}: no Transfermarkt club mapping — last XI unavailable.",
                                "tm_game_lineups",
                                f"{normalise(team)}:unmapped",
                            )
                        )
                        continue
                    last = self._q(
                        "SELECT g.game_id, g.date, g.home_club_id, g.home_club_name, g.away_club_name, g.competition_id, "
                        "CASE WHEN g.home_club_id = ? THEN g.home_club_formation ELSE g.away_club_formation END AS formation "
                        "FROM tm_games g WHERE (g.home_club_id = ? OR g.away_club_id = ?) AND g.date < ? ORDER BY g.date DESC LIMIT 1",
                        [cid, cid, cid, cutoff],
                    )
                    if last.empty:
                        claims.append(
                            self._missing(
                                f"{team}: no Transfermarkt game before {day(cutoff)}.",
                                "tm_games",
                                f"{normalise(team)}:none",
                            )
                        )
                        continue
                    g = last.iloc[0]
                    xi = self._q(
                        "SELECT player_name, position FROM tm_game_lineups WHERE game_id = ? AND club_id = ? AND type = 'starting_lineup' ORDER BY number NULLS LAST",
                        [g["game_id"], cid],
                    )
                    was_home = int(g["home_club_id"]) == int(cid)
                    opp = g["away_club_name"] if was_home else g["home_club_name"]
                    vals = {
                        "team": team,
                        "date": day(g["date"]),
                        "competition": str(g["competition_id"]),
                        "formation": str(g["formation"] or "n/a"),
                        "n": intg(len(xi)),
                        "xi": ", ".join(xi["player_name"].astype(str).tolist()) or "n/a",
                        "opponent": str(opp),
                        "label": "PROVISIONAL — last starting XI used, not a confirmed team sheet",
                    }
                    claims.append(
                        self._fact(
                            "tm_game_lineups",
                            f"{normalise(team)}:{vals['date']}",
                            vals,
                            "{label}: {team} started {n} players on {date} ({competition}, {formation} vs {opponent}): {xi}.",
                            status="provisional",
                        )
                    )
        # injuries / absences
        inj = pd.DataFrame()
        if self._has("api_football_injuries"):
            inj = self._q(
                "SELECT * FROM api_football_injuries WHERE team IN (?, ?) ORDER BY date DESC LIMIT 20",
                [fx.home_team, fx.away_team],
            )
        if not inj.empty:
            latest = pd.to_datetime(inj["date"]).max()
            self.coverage["injury_report_age_hours"] = round(
                float(
                    (
                        self.now - pd.Timestamp(latest).tz_localize(None)
                        if pd.Timestamp(latest).tzinfo
                        else self.now - pd.Timestamp(latest)
                    ).total_seconds()
                    / 3600
                ),
                1,
            )
            for team in (fx.home_team, fx.away_team):
                t = inj[inj["team"] == team]
                if t.empty:
                    continue
                vals = {
                    "team": team,
                    "n": intg(len(t)),
                    "items": "; ".join(
                        f"{p} ({r or t_})"
                        for p, r, t_ in zip(
                            t["player"], t.get("reason", [""] * len(t)), t.get("type", [""] * len(t)), strict=False
                        )
                    ),
                    "as_of": stamp(latest),
                }
                claims.append(
                    self._fact(
                        "api_football_injuries",
                        f"{fx.fixture_key}:{normalise(team)}",
                        vals,
                        "Injuries/suspensions {team} (API-Football, {as_of}): {n} listed — {items}.",
                    )
                )
        else:
            self.coverage["injury_report_age_hours"] = None
        # rotation load from Transfermarkt context
        far_ahead = fx.date is not None and (pd.Timestamp(fx.date).normalize() - self.now.normalize()).days > 10
        tm_end = None
        if self._has("tm_games"):
            tm_max = self._q("SELECT max(date) AS d FROM tm_games")
            if not tm_max.empty and pd.notna(tm_max["d"].iloc[0]):
                tm_end = pd.Timestamp(tm_max["d"].iloc[0])
        data_reaches = (
            tm_end is not None and fx.date is not None and tm_end >= pd.Timestamp(fx.date) - pd.Timedelta(days=8)
        )
        if tm_end is not None and fx.date is not None and not far_ahead and not data_reaches:
            claims.append(
                self._fact(
                    "tm_rotation_context",
                    f"{fx.fixture_key}:stale",
                    {"tm_end": day(tm_end), "date": day(fx.date), "window": "7"},
                    "Rotation load not computable: the open Transfermarkt extract ends {tm_end}, before the {window}-day window ahead of {date}.",
                    status="missing",
                )
            )
        if (
            self._has("tm_games")
            and self._has("tm_appearances")
            and fx.date is not None
            and not far_ahead
            and data_reaches
        ):
            from pitch_edge.features.context import transfermarkt_context

            one = pd.DataFrame(
                [
                    {
                        "match_id": fx.match_id or fx.fixture_key,
                        "date": pd.Timestamp(fx.date),
                        "home_team": fx.home_team,
                        "away_team": fx.away_team,
                    }
                ]
            )
            tmc = transfermarkt_context(self.wh, one)
            if not tmc.empty:
                r = tmc.iloc[0]
                vals = {
                    "home": fx.home_team,
                    "away": fx.away_team,
                    "h_min": num(r["rot_home_minutes_7d"], 0),
                    "a_min": num(r["rot_away_minutes_7d"], 0),
                    "h_days": intg(r["rot_home_days_since_any"]),
                    "a_days": intg(r["rot_away_days_since_any"]),
                    "h_cup": intg(r["rot_home_midweek_cup"]),
                    "a_cup": intg(r["rot_away_midweek_cup"]),
                    "window": "7",
                    "mw": "2-4",
                }
                claims.append(
                    self._fact(
                        "tm_rotation_context",
                        fx.fixture_key,
                        vals,
                        "Rotation load (Transfermarkt, all competitions): {home} {h_min} squad-minutes per starter in the last {window} days, last match {h_days} days ago, {h_cup} non-league fixture(s) {mw} days before; "
                        "{away} {a_min} minutes, {a_days} days ago, {a_cup} non-league fixture(s).",
                        status="ok" if fx.played else "provisional",
                    )
                )
        if self._has("tm_games") and fx.date is not None:
            tm_max = self._q("SELECT max(date) AS d FROM tm_games")
            if not tm_max.empty and pd.notna(tm_max["d"].iloc[0]):
                ref = min(pd.Timestamp(fx.date), self.now)  # age relative to today for future fixtures
                self.coverage["player_data_age_days"] = int((ref - pd.Timestamp(tm_max["d"].iloc[0])).days)
        return claims

    # ----------------------------------------------------------- section 4
    def _sec_referee(self, ctx: dict) -> list[Claim]:
        fx: Fixture = ctx["fx"]
        claims: list[Claim] = []
        referee, source = None, None
        if self._has("referee_announcements"):
            tk = fx.fixture_key.rsplit("|", 1)[0]
            ann = self._q("SELECT * FROM referee_announcements ORDER BY announced_at DESC")
            if not ann.empty:
                ann = ann[ann["fixture_key"].astype(str).str.rsplit("|", n=1).str[0] == tk]
                if fx.date is not None and "match_date" in ann:
                    md = pd.to_datetime(ann["match_date"], errors="coerce")
                    ann = ann[md.isna() | (md.dt.normalize() == pd.Timestamp(fx.date).normalize())]
            if not ann.empty:
                a = ann.iloc[0]
                referee, source = (
                    str(a["referee"]),
                    f"announced {stamp(a['announced_at'])} via {a.get('source', 'feed')}",
                )
        if referee is None and fx.match_id and self._has("features"):
            f = self._q("SELECT referee FROM features WHERE match_id = ?", [fx.match_id])
            if not f.empty and "referee" in f and pd.notna(f["referee"].iloc[0]) and str(f["referee"].iloc[0]):
                referee, source = str(f["referee"].iloc[0]), "match record"
        self.coverage["referee_assigned"] = referee is not None
        tend = self._referee_tendencies()
        if referee is None:
            vals = {
                "as_of": stamp(self.now),
                "n_refs": intg(len(tend)),
                "league_cpg": num(tend["cards_per_game"].mean(), 2) if not tend.empty else "n/a",
            }
            claims.append(
                self._fact(
                    "referee_tendency",
                    "unassigned",
                    vals,
                    "Referee not announced in our feeds as of {as_of}; the average of the {n_refs} referees with tendency data is {league_cpg} cards per game.",
                    status="missing",
                )
            )
            return claims
        row = tend[tend["referee"] == referee] if not tend.empty else pd.DataFrame()
        if row.empty:
            claims.append(
                self._fact(
                    "referee_announcements",
                    f"{fx.fixture_key}:{normalise(referee)}",
                    {"referee": referee, "source": source},
                    "Referee {referee} ({source}) — no tendency history in the warehouse yet.",
                    status="provisional",
                )
            )
            return claims
        r = row.iloc[0]
        vals = {
            "referee": referee,
            "source": str(source),
            "matches": intg(r["matches"]),
            "cpg": num(r["cards_per_game"], 2),
            "home_share": pct(r["home_card_share"]),
            "z_cards": snum(r["z_cards"], 2),
            "z_bias": snum(r["z_home_bias"], 2),
            "league_cpg": num(tend["cards_per_game"].mean(), 2),
            "tendency": "strong" if max(abs(float(r["z_cards"])), abs(float(r["z_home_bias"]))) >= 1 else "ordinary",
        }
        claims.append(
            self._fact(
                "referee_tendency",
                normalise(referee),
                vals,
                "Referee {referee} ({source}): {matches} matches on record, {cpg} cards/game (all referees {league_cpg}; z {z_cards}), home share of cards {home_share} (home-bias z {z_bias}) — {tendency} tendency.",
            )
        )
        return claims

    def _referee_tendencies(self) -> pd.DataFrame:
        from pitch_edge.backtest.event_study import referee_tendency_table

        if self._has("referee_tendency"):
            t = self.wh.read("referee_tendency")
            if not t.empty:
                return t
        if self._has("features") and "referee" in self.wh.columns("features"):
            f = self._q(
                "SELECT referee, date, home_yellows, away_yellows, home_reds, away_reds FROM features WHERE referee IS NOT NULL"
            )
            return referee_tendency_table(f)
        return pd.DataFrame()

    # ----------------------------------------------------------- section 5
    def _sec_conditions(self, ctx: dict) -> list[Claim]:
        fx: Fixture = ctx["fx"]
        claims: list[Claim] = []
        if fx.date is None:
            return [
                self._missing("No fixture date — rest, travel and weather cannot be assessed.", "conditions", "nodate")
            ]
        d = pd.Timestamp(fx.date)
        days_ahead = (d.normalize() - self.now.normalize()).days
        if days_ahead > 10:
            claims.append(
                self._fact(
                    "conditions",
                    f"{fx.fixture_key}:ahead",
                    {"days": intg(days_ahead), "horizon": "10"},
                    "Fixture is {days} days away (more than {horizon}): rest, congestion and kickoff weather cannot be known yet; travel distance below is fixed.",
                    status="missing",
                )
            )
            km = None
            if self._has("venues"):
                v = self._q("SELECT team, lat, lon FROM venues WHERE team IN (?, ?)", [fx.home_team, fx.away_team])
                if len(v) == 2:
                    from pitch_edge.data.alt.travel import haversine_km

                    h, a = v[v["team"] == fx.home_team].iloc[0], v[v["team"] == fx.away_team].iloc[0]
                    km = haversine_km(h["lat"], h["lon"], a["lat"], a["lon"])
            claims.append(
                self._fact(
                    "venues",
                    fx.fixture_key,
                    {"away": fx.away_team, "km": num(km, 0)},
                    "Away trip {away}: {km} km between home grounds.",
                )
            )
            self.coverage["weather_available"] = None
            return claims
        rest = {}
        for team in (fx.home_team, fx.away_team):
            last = self._q(
                "SELECT max(date) AS d, count(*) FILTER (WHERE date >= ?) AS n14 FROM matches WHERE (home_team = ? OR away_team = ?) AND date < ?",
                [d - pd.Timedelta(days=14), team, team, d],
            )
            rest[team] = (
                int((d - pd.Timestamp(last["d"].iloc[0])).days)
                if not last.empty and pd.notna(last["d"].iloc[0])
                else None,
                int(last["n14"].iloc[0]) if not last.empty else 0,
            )
        km = None
        if self._has("venues"):
            v = self._q("SELECT team, lat, lon FROM venues WHERE team IN (?, ?)", [fx.home_team, fx.away_team])
            if len(v) == 2:
                from pitch_edge.data.alt.travel import haversine_km

                h, a = v[v["team"] == fx.home_team].iloc[0], v[v["team"] == fx.away_team].iloc[0]
                km = haversine_km(h["lat"], h["lon"], a["lat"], a["lon"])
        vals = {
            "home": fx.home_team,
            "away": fx.away_team,
            "h_rest": intg(rest[fx.home_team][0]),
            "a_rest": intg(rest[fx.away_team][0]),
            "h_n14": intg(rest[fx.home_team][1]),
            "a_n14": intg(rest[fx.away_team][1]),
            "km": num(km, 0),
            "w": "14",
            "source": "league fixtures in the warehouse (spine) — cup/European games are in the rotation-load line above",
        }
        claims.append(
            self._fact(
                "matches_rest",
                fx.fixture_key,
                vals,
                "Rest: {home} {h_rest} days since their last league match ({h_n14} in the last {w} days); {away} {a_rest} days ({a_n14} in {w}); away trip {km} km between home grounds ({source}).",
            )
        )
        wx = None
        kind = None
        if fx.match_id and self._has("weather"):
            w = self._q("SELECT * FROM weather WHERE match_id = ?", [fx.match_id])
            if not w.empty:
                wx, kind = w.iloc[0].to_dict(), "Open-Meteo archive at the home ground"
        if (
            wx is None
            and self.fetch
            and not fx.played
            and 0 <= (d - self.now.normalize()).days <= 16
            and self._has("venues")
        ):
            v = self._q("SELECT lat, lon FROM venues WHERE team = ?", [fx.home_team])
            if not v.empty:
                from pitch_edge.data.alt.weather import OpenMeteoWeather

                hour = int(str(fx.kickoff_time or "15:00").split(":")[0]) if fx.kickoff_time else 15
                wx = OpenMeteoWeather().forecast_at(
                    float(v["lat"].iloc[0]), float(v["lon"].iloc[0]), d + pd.Timedelta(hours=hour)
                )
                kind = "Open-Meteo forecast for the kickoff hour" if wx else None
        self.coverage["weather_available"] = kind
        if wx:
            vals = {
                "kind": str(kind),
                "temp": num(wx.get("wx_temperature_2m"), 1),
                "rain": num(wx.get("wx_precipitation"), 1),
                "wind": num(wx.get("wx_wind_speed_10m"), 1),
            }
            claims.append(
                self._fact(
                    "weather",
                    fx.match_id or fx.fixture_key,
                    vals,
                    "Weather ({kind}): {temp} °C, {rain} mm precipitation, wind {wind} km/h.",
                    status="ok" if fx.played else "provisional",
                )
            )
        else:
            claims.append(
                self._missing(
                    "Kickoff weather unavailable (no archive row and no forecast within range).",
                    "weather",
                    fx.match_id or fx.fixture_key,
                )
            )
        return claims

    # ----------------------------------------------------------- section 6
    def _sec_market(self, ctx: dict) -> list[Claim]:
        fx: Fixture = ctx["fx"]
        claims: list[Claim] = []
        n_snap, venues = 0, 0
        if self._has("market_snapshots"):
            from pitch_edge.odds.leadlag import align_snapshots, snapshots_for_fixture

            snaps = self.wh.read("market_snapshots")
            long = align_snapshots(snaps, freq="1min") if not snaps.empty else pd.DataFrame()
            if not long.empty:
                long = snapshots_for_fixture(long, fx.home_team, fx.away_team)
            if not long.empty:
                self.market_series = [
                    {"venue": r.venue, "outcome": r.outcome, "ts": r.ts.isoformat(), "prob": float(r.prob)}
                    for r in long.itertuples()
                ]
                for venue, g in long.groupby("venue"):
                    venues += 1
                    n_snap += int(g["ts"].nunique())
                    vals = {
                        "venue": str(venue),
                        "n": intg(g["ts"].nunique()),
                        "first": stamp(g["ts"].min()),
                        "last": stamp(g["ts"].max()),
                    }
                    for o in OUTCOMES:
                        s = g[g["outcome"] == o].sort_values("ts")
                        vals[f"{o}_first"] = pct(s["prob"].iloc[0]) if len(s) else "n/a"
                        vals[f"{o}_last"] = pct(s["prob"].iloc[-1]) if len(s) else "n/a"
                        vals[f"{o}_move"] = spct(s["prob"].iloc[-1] - s["prob"].iloc[0]) if len(s) > 1 else "n/a"
                    claims.append(
                        self._fact(
                            "market_snapshots",
                            f"{fx.fixture_key}:{venue}",
                            vals,
                            "{venue}: {n} snapshot(s) {first} → {last}; home {home_first} → {home_last} ({home_move}), draw {draw_first} → {draw_last} ({draw_move}), away {away_first} → {away_last} ({away_move}).",
                        )
                    )
                if venues >= 2:
                    last = (
                        long.sort_values("ts")
                        .groupby(["venue", "outcome"])
                        .tail(1)
                        .pivot_table(index="outcome", columns="venue", values="prob")
                    )
                    if last.shape[1] >= 2:
                        gap = (last.iloc[:, 0] - last.iloc[:, 1]).abs()
                        o = str(gap.idxmax())
                        vals = {
                            "a": str(last.columns[0]),
                            "b": str(last.columns[1]),
                            "outcome": o,
                            "gap": pct(gap.max()),
                            "pa": pct(last.loc[o].iloc[0]),
                            "pb": pct(last.loc[o].iloc[1]),
                        }
                        claims.append(
                            self._fact(
                                "market_divergence",
                                fx.fixture_key,
                                vals,
                                "Cross-venue divergence: largest gap on {outcome}, {a} {pa} vs {b} {pb} ({gap}).",
                            )
                        )
        self.coverage["market_snapshots_n"], self.coverage["venues_quoting"] = n_snap, venues
        early = closing = False
        if fx.match_id and self._has("odds"):
            o = self._q(
                "SELECT bookmaker, side, price, is_closing FROM odds WHERE match_id = ? AND market = '1x2' AND bookmaker IN ('PS', 'Avg', 'Mkt', 'B365')",
                [fx.match_id],
            )
            if not o.empty:
                for book in ("PS", "Avg", "Mkt", "B365"):
                    b = o[o["bookmaker"] == book]
                    if b.empty:
                        continue
                    e = b[~b["is_closing"]].set_index("side")["price"]
                    c = b[b["is_closing"]].set_index("side")["price"]
                    if len(e) == 3:
                        early = True
                    if len(c) == 3:
                        closing = True
                    vals = {
                        "book": book,
                        **{f"e_{s}": num(e.get(s), 2) for s in OUTCOMES},
                        **{f"c_{s}": num(c.get(s), 2) for s in OUTCOMES},
                    }
                    claims.append(
                        self._fact(
                            "odds",
                            f"{fx.match_id}:{book}",
                            vals,
                            "{book} 1X2: early {e_home} / {e_draw} / {e_away} → closing {c_home} / {c_draw} / {c_away}.",
                        )
                    )
                    break
        self.coverage["early_price_available"] = early
        self.coverage["closing_price_available"] = closing
        if not claims:
            claims.append(
                self._missing(
                    "No price history for this fixture: no prediction-market snapshot matched and no bookmaker odds are stored (a live odds key would add real pre-match prices).",
                    "market_snapshots",
                    fx.fixture_key,
                )
            )
        return claims

    # ----------------------------------------------------------- section 7
    def _sec_narrative(self, ctx: dict) -> list[Claim]:
        fx: Fixture = ctx["fx"]
        as_of: pd.Timestamp = ctx["as_of"]
        claims: list[Claim] = []
        n_news = 0
        if self._has("news_items"):
            news = self._q(
                "SELECT * FROM news_items WHERE team IN (?, ?) AND published_at <= ? AND published_at > ? ORDER BY published_at DESC",
                [fx.home_team, fx.away_team, as_of, as_of - pd.Timedelta(hours=72)],
            )
            n_news = int(len(news))
            if not news.empty:
                from pitch_edge.data.alt.news import sentiment_velocity

                for r in news.head(6).itertuples():
                    did = doc_id("news", r.item_id)
                    self._docs[did] = Document(
                        did,
                        f"{r.title}. {r.summary or ''}".strip(),
                        {"type": "news", "feed": str(r.feed), "published_at": str(r.published_at), "team": str(r.team)},
                    )
                    tag = (
                        "INJURY"
                        if getattr(r, "is_injury_news", False)
                        else ("LINEUP" if getattr(r, "is_lineup_news", False) else "news")
                    )
                    vals = {
                        "when": stamp(r.published_at),
                        "feed": str(r.feed),
                        "lang": str(getattr(r, "language", None) or "en"),
                        "team": str(r.team),
                        "tag": tag,
                        "sent": snum(r.sentiment, 2),
                        "title": str(r.title),
                    }
                    claims.append(
                        self._fact(
                            "news_items",
                            str(r.item_id),
                            vals,
                            "{when} [{feed}, {lang}, {tag}] {team}: {title} (sentiment {sent}).",
                            extra_sources=[did],
                        )
                    )
                vel = sentiment_velocity(news)
                for r in vel.itertuples():
                    vals = {
                        "team": str(r.team),
                        "n": intg(r.n_items),
                        "mean": snum(r.sent_mean, 2),
                        "vel": snum(r.sent_velocity, 2),
                        "inj": intg(r.injury_items),
                        "w": "48",
                    }
                    claims.append(
                        self._fact(
                            "sentiment_velocity",
                            f"{fx.fixture_key}:{normalise(str(r.team))}",
                            vals,
                            "Sentiment velocity {team}: {n} items in the last {w}h, mean {mean}, change vs prior window {vel}, {inj} injury item(s).",
                        )
                    )
        self.coverage["news_items_72h"] = n_news
        if n_news == 0:
            claims.append(
                self._missing(
                    f"No news item linked to either club in the 72h before {stamp(as_of)} (EN + SV RSS feeds).",
                    "news_items",
                    f"{fx.fixture_key}:none",
                    {"as_of": stamp(as_of), "window_h": "72"},
                )
            )
        pv_age = None
        if self._has("wiki_pageviews"):
            from pitch_edge.data.alt.wikipedia_attention import attention_anomaly

            pv = self._q(
                "SELECT team, article, date, views FROM wiki_pageviews WHERE team IN (?, ?) AND date <= ? AND date >= ?",
                [fx.home_team, fx.away_team, as_of, as_of - pd.Timedelta(days=60)],
            )
            if not pv.empty:
                an = attention_anomaly(pv)
                pv_age = int((as_of.normalize() - pd.Timestamp(pv["date"].max())).days)
                for team in (fx.home_team, fx.away_team):
                    t = an[(an["team"] == team) & (pd.to_datetime(an["date"]) <= as_of.normalize())]
                    if t.empty:
                        continue
                    r = t.sort_values("date").iloc[-1]
                    art = str(pv[pv["team"] == team]["article"].iloc[0])
                    vals = {
                        "team": team,
                        "article": art,
                        "views": intg(r["pv_views"]),
                        "base": intg(r["pv_baseline"]),
                        "z": snum(r["pv_z"], 1),
                        "date": day(pd.Timestamp(r["date"]) - pd.Timedelta(days=1)),
                        "verdict": "ATTENTION SPIKE"
                        if float(r["pv_z"]) >= 3
                        else ("elevated" if float(r["pv_z"]) >= 1.5 else "normal"),
                        "w": "28",
                    }
                    claims.append(
                        self._fact(
                            "wiki_pageviews",
                            f"{normalise(team)}:{vals['date']}",
                            vals,
                            "Wikipedia attention {team} ({article}): {views} views on {date} vs {w}-day baseline {base} (robust z {z}) — {verdict}.",
                        )
                    )
        self.coverage["pageview_data_age_days"] = pv_age
        return claims

    # ----------------------------------------------------------- section 8
    def _sec_similar(self, ctx: dict) -> list[Claim]:
        fx: Fixture = ctx["fx"]
        if not self._has("features"):
            return [self._missing("No feature store — similar-match search unavailable.", "features", "none")]
        cols = ["elo_diff", "form_diff_r5", "form_diff_r10", "attack_diff_r5", "defence_diff_r5", "elo_exp_home"]
        cutoff = pd.Timestamp(fx.date) if fx.date is not None else self.now
        hist = self._q(
            f"SELECT match_id, date, league_code, home_team, away_team, home_goals, away_goals, {', '.join(cols)} FROM features WHERE date < ?",
            [cutoff],
        )
        hist = hist.dropna(subset=cols)
        if fx.match_id and fx.played:
            q = self._q(f"SELECT {', '.join(cols)} FROM features WHERE match_id = ?", [fx.match_id])
            hist = hist[hist["match_id"] != fx.match_id]
        else:
            q = self._carry_forward_query(fx, cols)
        if q is None or q.empty or q[cols].isna().any().any() or len(hist) < 50:
            return [
                self._missing(
                    "Not enough feature history to find similar matchups.", "features", f"{fx.fixture_key}:knn"
                )
            ]
        X = hist[cols].to_numpy(dtype=float)
        mu, sd = X.mean(axis=0), X.std(axis=0) + 1e-9
        Z = (X - mu) / sd
        zq = (q[cols].to_numpy(dtype=float)[0] - mu) / sd
        dist = np.sqrt(((Z - zq) ** 2).sum(axis=1))
        idx = np.argsort(dist)[:8]
        nn = hist.iloc[idx].assign(distance=dist[idx])
        hw = int((nn["home_goals"] > nn["away_goals"]).sum())
        dr = int((nn["home_goals"] == nn["away_goals"]).sum())
        claims = []
        vals = {
            "k": intg(len(nn)),
            "home_wins": intg(hw),
            "draws": intg(dr),
            "away_wins": intg(len(nn) - hw - dr),
            "space": ", ".join(cols),
            "note": "feature-space neighbours (standardised Euclidean); the GNN embeddings in this repo are player-level and are not a match embedding, so they are not used here",
        }
        claims.append(
            self._fact(
                "features_knn",
                fx.fixture_key,
                vals,
                "Of the {k} nearest historical profiles on ({space}): home won {home_wins}, drew {draws}, away won {away_wins} — {note}.",
            )
        )
        for r in nn.itertuples():
            did = doc_id("match", r.match_id)
            self._docs[did] = Document(
                did,
                f"{r.home_team} {intg(r.home_goals)}-{intg(r.away_goals)} {r.away_team} ({r.league_code}, {day(r.date)}); distance {num(r.distance, 2)}",
                {"type": "match", "match_id": str(r.match_id), "date": day(r.date)},
            )
            claims.append(
                Claim(
                    f"{day(r.date)} {r.home_team} {intg(r.home_goals)}-{intg(r.away_goals)} {r.away_team} ({r.league_code}; distance {num(r.distance, 2)})",
                    [did],
                )
            )
        return claims

    def _carry_forward_query(self, fx: Fixture, cols: list[str]) -> pd.DataFrame | None:
        h = self._team_rows(fx.home_team, self.now + pd.Timedelta(days=1), 1)
        a = self._team_rows(fx.away_team, self.now + pd.Timedelta(days=1), 1)
        if h.empty or a.empty:
            return None
        hr, ar = h.iloc[0], a.iloc[0]
        hp = "h_" if hr["home_team"] == fx.home_team else "a_"
        ap = "h_" if ar["home_team"] == fx.away_team else "a_"
        elo_h = hr["elo_home"] if hp == "h_" else hr["elo_away"]
        elo_a = ar["elo_home"] if ap == "h_" else ar["elo_away"]
        row = {
            "elo_diff": float(elo_h) - float(elo_a),
            "form_diff_r5": float(hr[f"{hp}pts_r5"]) - float(ar[f"{ap}pts_r5"]),
            "form_diff_r10": float(hr[f"{hp}pts_r10"]) - float(ar[f"{ap}pts_r10"]),
            "attack_diff_r5": float(hr[f"{hp}gf_r5"]) - float(ar[f"{ap}gf_r5"]),
            "defence_diff_r5": float(ar[f"{ap}ga_r5"]) - float(hr[f"{hp}ga_r5"]),
        }
        row["elo_exp_home"] = 1 / (1 + 10 ** (-(row["elo_diff"] + 60) / 400))
        return pd.DataFrame([row])[cols]

    # ----------------------------------------------------------- section 9
    def _sec_gaps(self, ctx: dict) -> list[Claim]:
        fx: Fixture = ctx["fx"]
        c = self.coverage
        gaps: list[str] = []
        if not c.get("confirmed_lineup"):
            gaps.append(
                "no confirmed XI yet"
                + (
                    " — the lineup shown is the last one used, not a team sheet"
                    if c.get("lineup_source") == "transfermarkt_last_xi"
                    else " and no lineup source is configured (API_FOOTBALL_KEY)"
                )
            )
        if not c.get("referee_assigned"):
            gaps.append(
                "referee not assigned/announced in our feeds — the referee-tendency term is at its league average"
            )
        if not c.get("early_price_available"):
            gaps.append(
                "no early bookmaker price for this fixture (Pinnacle early/closing arrive with football-data.co.uk; a live odds key adds soft-book prices)"
            )
        if c.get("injury_report_age_hours") is None:
            gaps.append("no structured injury report (only RSS keyword flags) — availability is inferred, not known")
        if not c.get("weather_available"):
            gaps.append("kickoff weather unknown")
        if (c.get("market_snapshots_n") or 0) == 0:
            gaps.append("no prediction-market snapshot matched this fixture, so market movement cannot be shown")
        elif (c.get("venues_quoting") or 0) < 2:
            gaps.append("only one venue quotes this fixture — no cross-venue divergence test possible")
        if c.get("model_prediction_source") == "fitted_at_dossier_time_raw":
            gaps.append(
                "the model probability is raw (fitted at dossier time, no fold calibration) — read it with the backtest evidence, not on its own"
            )
        elif c.get("model_prediction_source") is None:
            gaps.append("no model probability at all for this fixture")
        if (c.get("player_data_age_days") or 0) > 14:
            gaps.append("player/rotation data is stale relative to the fixture")
        if c.get("pageview_data_age_days") is None:
            gaps.append("no Wikipedia attention data for these clubs")
        if (c.get("news_items_72h") or 0) == 0:
            gaps.append("no linked news in the last 72h — silence, not confirmation")
        vals = {
            "generated_at": stamp(self.now),
            "fixture": fx.fixture_key,
            "confirmed_lineup": str(bool(c.get("confirmed_lineup"))),
            "lineup_source": str(c.get("lineup_source")),
            "referee_assigned": str(bool(c.get("referee_assigned"))),
            "early_price_available": str(bool(c.get("early_price_available"))),
            "closing_price_available": str(bool(c.get("closing_price_available"))),
            "market_snapshots_n": intg(c.get("market_snapshots_n") or 0),
            "venues_quoting": intg(c.get("venues_quoting") or 0),
            "injury_report_age_hours": num(c.get("injury_report_age_hours"), 1),
            "weather_available": str(c.get("weather_available")),
            "player_data_age_days": intg(c.get("player_data_age_days")),
            "pageview_data_age_days": intg(c.get("pageview_data_age_days")),
            "news_items_72h": intg(c.get("news_items_72h") or 0),
            "model_prediction_source": str(c.get("model_prediction_source")),
            "calibrated": str(bool(c.get("calibrated"))),
            "n_gaps": intg(len(gaps)),
            "window_h": "72",
        }
        claims = [
            self._fact(
                "intel_coverage",
                fx.fixture_key,
                vals,
                "Coverage as of {generated_at}: confirmed lineup {confirmed_lineup} ({lineup_source}); referee assigned {referee_assigned}; early price {early_price_available}; closing price {closing_price_available}; "
                "market snapshots {market_snapshots_n} across {venues_quoting} venue(s); injury report age {injury_report_age_hours} h; weather {weather_available}; player data age {player_data_age_days} d; "
                "pageview data age {pageview_data_age_days} d; news items in {window_h}h {news_items_72h}; prediction source {model_prediction_source} (calibrated {calibrated}). {n_gaps} gap(s) listed below.",
            )
        ]
        for i, g in enumerate(gaps, 1):
            claims.append(
                self._fact(
                    "intel_gap", f"{fx.fixture_key}:{i}", {"i": intg(i), "gap": g}, "Gap {i}: {gap}.", status="missing"
                )
            )
        if not gaps:
            claims.append(
                self._fact(
                    "intel_gap",
                    f"{fx.fixture_key}:0",
                    {"note": "every checklist item is present"},
                    "No known gaps: {note}.",
                )
            )
        return claims


# ------------------------------------------------------------------ persistence
def save_dossier(wh: Warehouse, dossier: Dossier) -> dict:
    ok, missing = dossier.verify()
    row = {
        "fixture_key": dossier.fixture_key,
        "version": dossier.version,
        "home_team": dossier.fixture.home_team,
        "away_team": dossier.fixture.away_team,
        "match_date": pd.Timestamp(dossier.fixture.date) if dossier.fixture.date is not None else pd.NaT,
        "league_code": dossier.fixture.league_code,
        "fixture_source": dossier.fixture.source,
        "model_name": dossier.model_name,
        "generated_at": dossier.generated_at,
        "verified": bool(ok),
        "unverified_numbers": ", ".join(missing),
        "n_claims": int(sum(len(s.claims) for s in dossier.sections)),
        "n_gaps": int(len([c for c in dossier.section("gaps").claims if c.status == "missing"])),
        "confirmed_lineup": bool(dossier.coverage.get("confirmed_lineup")),
        "markdown": dossier.markdown(),
        "json": dossier.to_json(),
        "content_hash": hashlib.sha1(dossier.markdown().encode()).hexdigest()[:12],
    }
    wh.upsert("dossiers", pd.DataFrame([row]))
    return row


def load_dossier_versions(wh: Warehouse, fixture_key: str) -> pd.DataFrame:
    if not wh.table_exists("dossiers"):
        return pd.DataFrame()
    return wh.query("SELECT * FROM dossiers WHERE fixture_key = ? ORDER BY version DESC", [fixture_key])


def diff_dossiers(previous_json: str | dict, current: Dossier | str | dict) -> dict:
    """Section-by-section claim diff between two dossier versions: {section: {'added': [...], 'removed': [...]}}."""
    prev = json.loads(previous_json) if isinstance(previous_json, str) else previous_json
    cur = (
        json.loads(current.to_json())
        if isinstance(current, Dossier)
        else (json.loads(current) if isinstance(current, str) else current)
    )
    out: dict = {"from_version": prev.get("version"), "to_version": cur.get("version"), "sections": {}}
    p_secs = {s["key"]: {c["text"] for c in s["claims"]} for s in prev.get("sections", [])}
    c_secs = {s["key"]: {c["text"] for c in s["claims"]} for s in cur.get("sections", [])}
    for key in dict.fromkeys([*p_secs, *c_secs]):
        added = sorted(c_secs.get(key, set()) - p_secs.get(key, set()))
        removed = sorted(p_secs.get(key, set()) - c_secs.get(key, set()))
        if added or removed:
            out["sections"][key] = {"added": added, "removed": removed}
    out["changed"] = bool(out["sections"])
    return out


def render_diff(diff: dict) -> str:
    if not diff.get("changed"):
        return f"No change since version {diff.get('from_version')}."
    lines = [f"What changed since v{diff.get('from_version')} → v{diff.get('to_version')}:"]
    for key, d in diff["sections"].items():
        lines.append(f"  [{key}]")
        lines += [f"    + {t}" for t in d["added"]]
        lines += [f"    - {t}" for t in d["removed"]]
    return "\n".join(lines)
