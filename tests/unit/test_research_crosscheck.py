"""Cross-check `pitch_edge.research` against the independent interim implementation the R2 study wrote
(`research/studies/r2-common/r2lib.py`). Two codebases written separately agreeing on random inputs is
evidence neither has a transcription error. Skipped when the study file is not present."""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from pitch_edge.research import devig, stats

pytestmark = pytest.mark.unit

R2LIB = Path(__file__).resolve().parents[2] / "research" / "studies" / "r2-common" / "r2lib.py"


@pytest.fixture(scope="module")
def r2() -> ModuleType:
    if not R2LIB.exists():
        pytest.skip("research/studies/r2-common/r2lib.py not present")
    spec = importlib.util.spec_from_file_location("r2lib_crosscheck", R2LIB)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _books(seed: int, n: int = 200) -> np.ndarray:
    rng = np.random.default_rng(seed)
    fair = rng.dirichlet([3.0, 1.5, 2.0], size=n)
    margin = rng.uniform(0.01, 0.10, size=(n, 1))
    return 1.0 / (fair * (1 + margin))


@pytest.mark.parametrize("method", ["multiplicative", "power", "shin"])
def test_devig_agrees(r2, method):
    odds = _books(1)
    np.testing.assert_allclose(devig.devig(odds, method).probs, r2.DEVIG[method](odds), atol=1e-9)


def test_bh_agrees(r2):
    p = np.random.default_rng(2).uniform(size=60) ** 2
    p[5] = np.nan
    _, q = stats.benjamini_hochberg(p, q=0.10)
    q_r2, _ = r2.bh(p, q=0.10)
    np.testing.assert_allclose(q, q_r2, equal_nan=True)


@pytest.mark.parametrize(
    ("sr", "n", "skew", "kurt", "trials", "var"), [(0.1, 500, -0.4, 4.5, 30, 0.002), (0.05, 2000, 0.2, 3.1, 7, 0.0005)]
)
def test_psr_dsr_agree(r2, sr, n, skew, kurt, trials, var):
    assert stats.probabilistic_sharpe_ratio(sr, n, skew, kurt, 0.02) == pytest.approx(r2.psr(sr, 0.02, n, skew, kurt))
    assert stats.deflated_sharpe_ratio(sr, n, skew, kurt, trials, var) == pytest.approx(
        r2.dsr(sr, n, skew, kurt, trials, var), abs=1e-9
    )


def test_pbo_agrees(r2):
    rng = np.random.default_rng(3)
    m = rng.normal(size=(320, 6))
    m[:, 2] += 0.15
    ours = stats.pbo_cscv(m, n_splits=8).pbo
    assert math.isclose(ours, r2.pbo_cscv(m, n_blocks=8), abs_tol=1e-12)
