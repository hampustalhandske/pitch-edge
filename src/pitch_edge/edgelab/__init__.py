"""Bridge from pitch-edge to the edgelab grading engine: dataset export and strategy scaffolding.

Nothing here imports edgelab; the package only writes files edgelab's `parquet` data provider and
strategy loader read.
"""

from pitch_edge.edgelab.export import (
    MARKET_TYPE_CODES,
    MODEL_FEATURE,
    EdgelabExportError,
    ExportReport,
    export_prediction_market_dataset,
)
from pitch_edge.edgelab.scaffold import ScaffoldSpec, scaffold_strategy

__all__ = [
    "MARKET_TYPE_CODES",
    "MODEL_FEATURE",
    "EdgelabExportError",
    "ExportReport",
    "ScaffoldSpec",
    "export_prediction_market_dataset",
    "scaffold_strategy",
]
