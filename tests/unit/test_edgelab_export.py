"""Tests for the edgelab dataset exporter and strategy scaffold (no edgelab import)."""

import json
import os
import shutil
import subprocess
from pathlib import Path

import pandas as pd
import pytest

from pitch_edge.edgelab import (
    MODEL_FEATURE,
    EdgelabExportError,
    ScaffoldSpec,
    export_prediction_market_dataset,
    scaffold_strategy,
)
from pitch_edge.edgelab.scaffold import ScaffoldError

pytestmark = pytest.mark.unit

KICK = pd.Timestamp("2025-03-01 15:00", tz="UTC")


def _inputs(n_markets: int = 3):
    snaps, res, feats = [], [], []
    for i in range(n_markets):
        market = f"epl-m{i:02d}-win_home"
        kick = KICK + pd.Timedelta(days=i)
        for h in range(6, 1, -1):
            snaps.append(
                {"market": market, "ts": kick - pd.Timedelta(hours=h), "price": 0.4, "spread": 0.02, "size": 500.0}
            )
        res.append({"market": market, "resolved_at": kick + pd.Timedelta(hours=3), "outcome": i % 2})
        feats.append(
            {
                "market": market,
                "ts": kick - pd.Timedelta(hours=5),
                "known_at": kick - pd.Timedelta(hours=4),
                "model_p": 0.5,
                "kickoff_ts": kick,
                "market_type": "win_home",
            }
        )
    return pd.DataFrame(snaps), pd.DataFrame(res), pd.DataFrame(feats)


def test_export_writes_valid_dataset(tmp_path: Path) -> None:
    s, r, f = _inputs()
    rep = export_prediction_market_dataset(s, r, f, tmp_path / "d", source="synthetic")
    assert (rep.n_markets, rep.n_snapshots, rep.n_resolutions) == (3, 15, 3)
    book = pd.read_parquet(tmp_path / "d" / "book_snapshots.parquet")
    assert list(book.columns) == ["ts", "market", "level", "bid_price", "bid_size", "ask_price", "ask_size", "source"]
    assert str(book["ts"].dtype) == "datetime64[ns, UTC]"
    assert book["bid_price"].iloc[0] == pytest.approx(0.39)
    assert book["ask_price"].iloc[0] == pytest.approx(0.41)
    feats = pd.read_parquet(tmp_path / "d" / "features.parquet")
    assert set(feats["name"]) == {"kickoff_ts", "market_type", MODEL_FEATURE}
    k = feats[(feats.symbol == "epl-m00-win_home") & (feats.name == "kickoff_ts")]
    assert k["value"].iloc[0] == KICK.timestamp()
    assert (feats["known_at"] >= feats["ts"]).all()
    assert len(pd.read_parquet(tmp_path / "d" / "bars.parquet")) == 0
    res = pd.read_parquet(tmp_path / "d" / "resolutions.parquet")
    assert res["outcome"].dtype == "int64"


def test_market_type_can_be_omitted(tmp_path: Path) -> None:
    s, r, f = _inputs()
    export_prediction_market_dataset(s, r, f.drop(columns=["market_type"]), tmp_path / "d", include_market_type=False)
    feats = pd.read_parquet(tmp_path / "d" / "features.parquet")
    assert set(feats["name"]) == {"kickoff_ts", MODEL_FEATURE}
    with pytest.raises(EdgelabExportError, match="market_type"):
        export_prediction_market_dataset(s, r, f.drop(columns=["market_type"]), tmp_path / "e")


def test_bid_ask_input_accepted(tmp_path: Path) -> None:
    s, r, f = _inputs(1)
    s = s.assign(bid=0.38, ask=0.42).drop(columns=["price", "spread"])
    export_prediction_market_dataset(s, r, f, tmp_path)


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda s, r, f: s.__setitem__("price", 0.0), "inside"),
        (lambda s, r, f: s.__setitem__("price", 0.995), "inside"),
        (lambda s, r, f: s.__setitem__("ts", s["ts"].dt.tz_localize(None)), "timezone"),
        (lambda s, r, f: r.__setitem__("outcome", 2), "0 or 1"),
        (lambda s, r, f: f.__setitem__("known_at", f["ts"] - pd.Timedelta(minutes=1)), "known_at earlier"),
        (lambda s, r, f: f.__setitem__("model_p", 1.0), "model_p"),
        (lambda s, r, f: r.__setitem__("resolved_at", s["ts"].min()), "last snapshot"),
        (lambda s, r, f: f.__setitem__("market_type", "nonsense"), "unknown"),
        (lambda s, r, f: s.__setitem__("spread", -0.1), "spread"),
    ],
)
def test_invariants_rejected(tmp_path: Path, mutate, match: str) -> None:
    s, r, f = _inputs()
    mutate(s, r, f)
    with pytest.raises(EdgelabExportError, match=match):
        export_prediction_market_dataset(s, r, f, tmp_path / "x")


def test_structural_rejections(tmp_path: Path) -> None:
    s, r, f = _inputs()
    with pytest.raises(EdgelabExportError, match="different markets"):
        export_prediction_market_dataset(s, r.iloc[:-1], f, tmp_path / "a")
    with pytest.raises(EdgelabExportError, match="duplicate"):
        export_prediction_market_dataset(pd.concat([s, s.iloc[:1]]), r, f, tmp_path / "b")
    late = r.assign(resolved_at=KICK - pd.Timedelta(days=10))
    with pytest.raises(EdgelabExportError, match="resolved_at before"):
        export_prediction_market_dataset(s, late, f, tmp_path / "c")
    in_play = pd.concat(
        [
            s,
            pd.DataFrame(
                [
                    {
                        "market": "epl-m00-win_home",
                        "ts": KICK + pd.Timedelta(hours=1),
                        "price": 0.9,
                        "spread": 0.02,
                        "size": 1.0,
                    }
                ]
            ),
        ]
    )
    with pytest.raises(EdgelabExportError, match="kickoff"):
        export_prediction_market_dataset(in_play, r, f, tmp_path / "d")
    export_prediction_market_dataset(in_play, r, f, tmp_path / "e", allow_in_play=True)
    odd = f.copy()
    odd.loc[odd.index[0], "kickoff_ts"] = KICK + pd.Timedelta(hours=1)
    odd = pd.concat(
        [
            odd,
            f.iloc[:1].assign(
                ts=f["ts"].iloc[0] + pd.Timedelta(minutes=5), known_at=f["known_at"].iloc[0] + pd.Timedelta(minutes=5)
            ),
        ]
    )
    with pytest.raises(EdgelabExportError, match="vary"):
        export_prediction_market_dataset(s, r, odd, tmp_path / "f")


def _spec(**kw) -> ScaffoldSpec:
    base = {
        "slug": "epl-model",
        "title": "Pitch-edge EPL model",
        "rationale": "The model beats the closing price.",
        "data_path": "/tmp/data",
        "range_start": "2024-01-01",
        "range_end": "2025-12-31",
    }
    base.update(kw)
    return ScaffoldSpec(**base)


def test_scaffold_files(tmp_path: Path) -> None:
    spec = _spec(edge_thresholds=(0.02, 0.05), selected={"edge_threshold": 0.05})
    folder = scaffold_strategy(spec, tmp_path)
    assert folder.name == "pitch-edge-epl-model"
    code = (folder / "strategy.py").read_text()
    compile(code, "strategy.py", "exec")
    assert "class PitchEdgeEplModel(PredictionMarketStrategy)" in code
    assert MODEL_FEATURE in code
    text = (folder / "strategy.yaml").read_text()
    assert "edge_threshold: [0.02, 0.05]" in text
    assert "selected: {edge_threshold: 0.05}" in text
    assert "kelly_fraction: 0.25" in text.split("parameters:")[0]
    assert "provider: parquet" in text and "grading:" not in text.replace("grading: block", "")
    assert spec.n_trials == 2
    assert json.loads((folder / "strategy.json").read_text())["asset_class"] == "prediction_market"
    yaml = pytest.importorskip("yaml")
    doc = yaml.safe_load(text)
    assert doc["parameters"]["grid"] == {"edge_threshold": [0.02, 0.05]}
    assert doc["capital"] == 100000.0 and doc["seed"] == 7
    assert doc["strategy"].endswith(":PitchEdgeEplModel")
    with pytest.raises(ScaffoldError, match="overwrite"):
        scaffold_strategy(spec, tmp_path)


@pytest.mark.parametrize(
    "kw",
    [
        {"slug": "Bad Slug"},
        {"rationale": " "},
        {"entry_hours_before_kickoff": 1.0},
        {"edge_thresholds": (0.02, 0.05)},
        {"edge_thresholds": (0.02, 0.05), "selected": {"edge_threshold": 0.9}},
        {"selected": {"kelly_fraction": 0.25}},
        {"kelly_fractions": (0.0,)},
    ],
)
def test_scaffold_rejects_bad_spec(tmp_path: Path, kw) -> None:
    with pytest.raises(ScaffoldError):
        scaffold_strategy(_spec(**kw), tmp_path)


EDGELAB_PY = Path(os.environ.get("EDGELAB_DIR", "/nonexistent")) / ".venv/bin/python"


@pytest.mark.integration
@pytest.mark.skipif(not EDGELAB_PY.exists(), reason="EDGELAB_DIR unset or its venv missing")
def test_edgelab_schema_validator_accepts_export(tmp_path: Path) -> None:
    s, r, f = _inputs()
    export_prediction_market_dataset(s, r, f, tmp_path / "d")
    code = (
        "import sys, pandas as pd\n"
        "from edgelab.data.schemas import BOOK_SNAPSHOTS, RESOLUTIONS, FEATURES, BARS, validate_frame\n"
        "d = sys.argv[1]\n"
        "for n, sch in (('bars', BARS), ('book_snapshots', BOOK_SNAPSHOTS), ('resolutions', RESOLUTIONS), ('features', FEATURES)):\n"
        "    validate_frame(pd.read_parquet(f'{d}/{n}.parquet'), sch)\n"
        "print('ok')\n"
    )
    arch = shutil.which("arch")
    cmd = (
        [arch, "-arm64", str(EDGELAB_PY), "-I", "-c", code, str(tmp_path / "d")]
        if arch
        else [str(EDGELAB_PY), "-I", "-c", code, str(tmp_path / "d")]
    )
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stderr
    assert "ok" in out.stdout
