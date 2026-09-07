"""StatsBomb events -> passing graphs -> GNN embeddings; events -> minute states -> in-play model."""

from __future__ import annotations

import numpy as np
import pytest

from pitch_edge.models.gnn import PlayerEmbeddingGNN, build_passing_graphs
from pitch_edge.models.inplay import STATE_COLS, InPlayWinProbabilityModel, minute_states

pytestmark = pytest.mark.integration


def test_passing_graphs_have_nodes_edges_and_xg(statsbomb_events_df):
    graphs = build_passing_graphs(statsbomb_events_df)
    assert len(graphs) == 2
    g = next(x for x in graphs if x.team == "Home FC")
    assert g.data.x.shape == (3, 13) and g.data.edge_index.shape[0] == 2 and g.data.edge_index.shape[1] > 0
    assert g.team_xg > 0 and g.opp_xg > 0


def test_gnn_trains_and_supports_similarity(statsbomb_events_df):
    graphs = build_passing_graphs(statsbomb_events_df)
    gnn = PlayerEmbeddingGNN(epochs=5, hidden=8, out=4).fit(graphs)
    emb = gnn.embeddings_
    assert len(emb) == 6 and {"e0", "e3", "n_matches"} <= set(emb.columns)
    sim = gnn.most_similar("H One", k=2)
    assert len(sim) == 2 and sim["similarity"].between(-1, 1).all()
    assert gnn.most_similar("Nobody").empty
    assert "GraphSAGE" in gnn.card()["family"]


def test_minute_states_and_inplay_model(statsbomb_events_df):
    states = minute_states(statsbomb_events_df, pre_exp_home=0.55, step=5)
    assert set(STATE_COLS) <= set(states.columns)
    assert states["score_diff"].iloc[-1] == 1  # home scored at 40'
    assert (states["final_result"] == 0).all()
    # two "matches" for training: the real one and a mirrored copy with a different id/result
    mirrored = states.copy()
    mirrored["statsbomb_match_id"] = 1000
    mirrored["score_diff"] *= -1
    mirrored["xg_diff"] *= -1
    mirrored["final_result"] = 2
    train = np.concatenate([states.to_numpy(), mirrored.to_numpy()])
    import pandas as pd

    train_df = pd.DataFrame(train, columns=states.columns).astype(states.dtypes.to_dict())
    model = InPlayWinProbabilityModel(epochs=5, hidden=8).fit(train_df)
    path = model.predict_path(states)
    assert np.allclose(path[["home", "draw", "away"]].sum(axis=1), 1.0, atol=1e-5)
    assert path["home"].iloc[-1] > path["away"].iloc[-1]  # leading at the end
    assert "no Betfair in-play line" in " ".join(model.card()["limitations"])
