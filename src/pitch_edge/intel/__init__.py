"""Match Intel — the pre-match dossier (Phase 5, Pillar A).

Every sentence is either a number read from a warehouse table / report artifact with its source
named, or a RAG document quoted with its `[doc_id]`; `Dossier.verify()` runs the same
`verify_citations` check as the RAG layer, so an ungrounded figure fails loudly instead of shipping.
"""

from pitch_edge.intel.dossier import Dossier, DossierBuilder, diff_dossiers, load_dossier_versions, save_dossier
from pitch_edge.intel.fixture import Fixture, find_fixture

__all__ = [
    "Dossier",
    "DossierBuilder",
    "Fixture",
    "diff_dossiers",
    "find_fixture",
    "load_dossier_versions",
    "save_dossier",
]
