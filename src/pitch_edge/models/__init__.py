from pitch_edge.models.base import OUTCOMES, MatchModel
from pitch_edge.models.calibration import IsotonicCalibrator, brier_score, log_loss_score, reliability_curve
from pitch_edge.models.dixon_coles import DixonColesModel
from pitch_edge.models.gbdt import GBDTMatchModel
from pitch_edge.models.poisson import DixonColesMatchModel
from pitch_edge.models.sequence import GRUSequenceModel, TransformerSequenceModel

__all__ = [
    "OUTCOMES",
    "DixonColesMatchModel",
    "DixonColesModel",
    "GBDTMatchModel",
    "GRUSequenceModel",
    "TransformerSequenceModel",
    "IsotonicCalibrator",
    "MatchModel",
    "brier_score",
    "log_loss_score",
    "reliability_curve",
]


def available_models() -> list[MatchModel]:
    """Every match-outcome model the CLI can select by name (superset of `default_models`)."""
    return [
        DixonColesMatchModel(),
        GBDTMatchModel(include_market=False),
        GBDTMatchModel(include_market=True),
        TransformerSequenceModel(),
        # Phase 6 candidate (CASE_STUDY.md Result 8): confirmed-lineup squad-value + missing-star-value
        # features on top of the same GBDT recipe as `gbdt`/`gbdt_mkt`. Only excludes the Result-7
        # rejected groups so the new `sv_` group is the sole difference from the baseline GBDTs.
        GBDTMatchModel(include_market=False, exclude_prefixes=("pv_", "rot_"), name_suffix="_squadval"),
        GBDTMatchModel(include_market=True, exclude_prefixes=("pv_", "rot_"), name_suffix="_squadval"),
    ]


def default_models(include_market: bool = True) -> list[MatchModel]:
    models: list[MatchModel] = [DixonColesMatchModel(), GBDTMatchModel(include_market=False), TransformerSequenceModel()]
    if include_market:
        models.append(GBDTMatchModel(include_market=True))
    return models
