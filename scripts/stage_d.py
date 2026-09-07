"""Stage D: rebuild RAG index (hybrid), retrieval eval, Lightning GRU + Transformer backtests (2018+ slice)."""

import logging
import shutil
import time

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
from pitch_edge.backtest.engine import WalkForwardConfig
from pitch_edge.config import get_settings
from pitch_edge.data.storage import Warehouse
from pitch_edge.models import GRUSequenceModel, TransformerSequenceModel
from pitch_edge.pipeline import build_rag_index, load_feature_frame, run_backtests
from pitch_edge.rag.documents import match_documents, news_documents, prediction_documents
from pitch_edge.rag.eval import evaluate_retrieval, summarize_eval, synthetic_questions
from pitch_edge.rag.index import VectorIndex

s = get_settings()
s.ensure_dirs()
t0 = time.time()
shutil.rmtree(s.vector_dir, ignore_errors=True)
with Warehouse(s.db_path) as wh:
    idx = build_rag_index(wh, VectorIndex())
    print("rag rebuilt:", idx.backend, idx.count(), "docs in", round(time.time() - t0), "s", flush=True)
    docs = match_documents(wh.matches_with_closing_odds(bookmakers=("PS", "Mkt")), limit=3000)
    preds = wh.read("model_predictions")
    latest = preds.sort_values("run_id")["run_id"].iloc[-1]
    for m, g in preds[preds["run_id"] == latest].groupby("model_name"):
        docs += prediction_documents(g.tail(500), str(m), None)
    docs += news_documents(wh.read("news_items"))
    res = evaluate_retrieval(idx, synthetic_questions(docs, n=80), k=5)
    res.to_csv(s.reports_dir / "rag_eval.csv", index=False)
    print(summarize_eval(res).round(3).to_string(index=False), flush=True)
    run_id = (s.artifacts_dir / "run_id.txt").read_text().strip()
    f = load_feature_frame(wh, min_date="2018-07-01")
    print("features", f.shape, flush=True)
    run_backtests(
        f,
        [GRUSequenceModel(epochs=20), TransformerSequenceModel(epochs=20)],
        WalkForwardConfig(retrain_every_days=90),
        wh=wh,
        label="main",
        run_id=run_id,
    )
    print("sequence backtests done", round(time.time() - t0), "s", flush=True)
print("STAGE D COMPLETE", round(time.time() - t0), "s", flush=True)
