"""Bridge from pitch-edge to the edgelab grading engine: dataset export, strategy scaffolding and the
submission manifest.

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
from pitch_edge.edgelab.submission import SubmissionError, SubmissionSpec, write_submission

__all__ = [
    "MARKET_TYPE_CODES",
    "MODEL_FEATURE",
    "EdgelabExportError",
    "ExportReport",
    "ScaffoldSpec",
    "SubmissionError",
    "SubmissionSpec",
    "export_prediction_market_dataset",
    "scaffold_strategy",
    "write_submission",
]
