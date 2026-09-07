from pitch_edge.models.base import OUTCOMES, MatchModel
from pitch_edge.models.calibration import IsotonicCalibrator, brier_score, log_loss_score, reliability_curve
from pitch_edge.models.dixon_coles import DixonColesModel
from pitch_edge.models.gbdt import GBDTMatchModel
from pitch_edge.models.gnn import PlayerEmbeddingGNN, build_passing_graphs
from pitch_edge.models.inplay import InPlayWinProbabilityModel, minute_states
from pitch_edge.models.poisson import DixonColesMatchModel
from pitch_edge.models.sequence import GRUSequenceModel, TransformerSequenceModel

__all__ = [
    "OUTCOMES",
    "DixonColesMatchModel",
    "DixonColesModel",
    "GBDTMatchModel",
    "GRUSequenceModel",
    "TransformerSequenceModel",
    "InPlayWinProbabilityModel",
    "IsotonicCalibrator",
    "MatchModel",
    "PlayerEmbeddingGNN",
    "brier_score",
    "build_passing_graphs",
    "log_loss_score",
    "minute_states",
    "reliability_curve",
]


def available_models() -> list[MatchModel]:
    """Every match-outcome model the CLI can select by name (superset of `default_models`)."""
    return [
        DixonColesMatchModel(),
        GBDTMatchModel(include_market=False),
        GBDTMatchModel(include_market=True),
        GRUSequenceModel(),
        TransformerSequenceModel(),
    ]


def default_models(include_market: bool = True) -> list[MatchModel]:
    models: list[MatchModel] = [DixonColesMatchModel(), GBDTMatchModel(include_market=False), GRUSequenceModel()]
    if include_market:
        models.append(GBDTMatchModel(include_market=True))
    return models
