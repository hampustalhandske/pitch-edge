from pitch_edge.models.base import OUTCOMES, MatchModel
from pitch_edge.models.calibration import IsotonicCalibrator, brier_score, log_loss_score, reliability_curve
from pitch_edge.models.dixon_coles import DixonColesModel
from pitch_edge.models.gbdt import GBDTMatchModel
from pitch_edge.models.poisson import DixonColesMatchModel
from pitch_edge.models.sentiment import SentimentOnlyModel
from pitch_edge.models.sequence import GRUSequenceModel, TransformerSequenceModel
from pitch_edge.models.stochastic import StochasticStrengthModel

__all__ = [
    "OUTCOMES",
    "DixonColesMatchModel",
    "DixonColesModel",
    "GBDTMatchModel",
    "GRUSequenceModel",
    "SentimentOnlyModel",
    "StochasticStrengthModel",
    "TransformerSequenceModel",
    "IsotonicCalibrator",
    "MatchModel",
    "brier_score",
    "log_loss_score",
    "reliability_curve",
]


def available_models() -> list[MatchModel]:
    """Every match-outcome model the CLI can select by name. Deliberately five *different*
    approaches rather than several GBDT variants: a goals-process model, one tree ensemble on the
    agreed feature set (no market odds — there's no live feed, so a historical market column would
    just be bias, never something the live `ask` path can see), a Monte Carlo stochastic-process
    model, a deep sequence model, and a sentiment-only model that isolates the news/LLM signal."""
    return [
        DixonColesMatchModel(),
        GBDTMatchModel(include_market=False),
        StochasticStrengthModel(),
        TransformerSequenceModel(),
        SentimentOnlyModel(),
    ]


def default_models() -> list[MatchModel]:
    """The fast-to-refit subset `ask` uses (no market odds, no sequence model — too slow to refit
    per question)."""
    return [DixonColesMatchModel(), GBDTMatchModel(include_market=False), StochasticStrengthModel()]
