"""GNN player embeddings from StatsBomb passing networks (PyTorch Geometric).

Each match-team is a directed, weighted passing graph (nodes = players who acted or received a
pass; edges = completed passes; node features = per-match event profile). A GraphSAGE or GAT
encoder is trained self-supervised on link prediction (held-out edges vs random negatives) plus a
light team-xG-share head. Link-prediction AUROC on the held-out edges is reported so the
embedding quality is measured, not asserted. Embeddings support "find a similar player" and
team-strength descriptors. Event-data only; no market data.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torchmetrics
from torch import nn
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, SAGEConv

NODE_FEATURES = [
    "passes",
    "pass_completion",
    "progressive_passes",
    "shots",
    "xg",
    "pressures",
    "carries",
    "dribbles",
    "ball_recoveries",
    "interceptions",
    "avg_x",
    "avg_y",
    "under_pressure_rate",
]


@dataclass
class PassingGraph:
    match_id: int
    team: str
    player_ids: list[int]
    player_names: list[str]
    data: Data
    team_xg: float
    opp_xg: float


def build_passing_graphs(events: pd.DataFrame) -> list[PassingGraph]:
    graphs: list[PassingGraph] = []
    if events.empty:
        return graphs
    for (mid, team), ev in events.groupby(["statsbomb_match_id", "team"]):
        ev = ev.dropna(subset=["player_id"])
        if ev.empty:
            continue
        acted = ev["player_id"].astype(int).tolist()
        received = ev.loc[ev["pass_recipient_id"].notna(), "pass_recipient_id"].astype(int).tolist()
        players = list(dict.fromkeys(acted + received))
        idx = {p: i for i, p in enumerate(players)}
        names = ev.drop_duplicates("player_id").set_index("player_id")["player"].to_dict()
        names.update(
            ev.loc[ev["pass_recipient_id"].notna()]
            .drop_duplicates("pass_recipient_id")
            .set_index("pass_recipient_id")["pass_recipient"]
            .to_dict()
        )

        passes = ev[ev["type"] == "Pass"]
        completed = passes[passes["pass_outcome"].isna() & passes["pass_recipient_id"].notna()]
        edge_counts: dict[tuple[int, int], int] = {}
        for src, dst in zip(
            completed["player_id"].astype(int), completed["pass_recipient_id"].astype(int), strict=True
        ):
            if dst in idx:
                edge_counts[(idx[src], idx[dst])] = edge_counts.get((idx[src], idx[dst]), 0) + 1
        if not edge_counts:
            continue
        ei = torch.tensor(list(edge_counts.keys()), dtype=torch.long).t().contiguous()
        ew = torch.tensor(list(edge_counts.values()), dtype=torch.float32)

        feats = np.zeros((len(players), len(NODE_FEATURES)), dtype=np.float32)
        for p, i in idx.items():
            pe = ev[ev["player_id"] == p]
            pp = pe[pe["type"] == "Pass"]
            n_pass = len(pp)
            comp = pp["pass_outcome"].isna().sum() if n_pass else 0
            prog = ((pp["pass_end_x"].fillna(0) - pp["x"].fillna(0)) > 10).sum() if n_pass else 0
            shots = pe[pe["type"] == "Shot"]
            feats[i] = [
                n_pass,
                comp / n_pass if n_pass else 0.0,
                prog,
                len(shots),
                float(shots["shot_xg"].fillna(0).sum()),
                (pe["type"] == "Pressure").sum(),
                (pe["type"] == "Carry").sum(),
                (pe["type"] == "Dribble").sum(),
                (pe["type"] == "Ball Recovery").sum(),
                (pe["type"] == "Interception").sum(),
                float(pe["x"].mean() if pe["x"].notna().any() else 60.0),
                float(pe["y"].mean() if pe["y"].notna().any() else 40.0),
                float(pe["under_pressure"].mean()) if len(pe) else 0.0,
            ]
        team_xg = float(ev.loc[ev["type"] == "Shot", "shot_xg"].fillna(0).sum())
        opp = events[(events["statsbomb_match_id"] == mid) & (events["team"] != team) & (events["type"] == "Shot")]
        graphs.append(
            PassingGraph(
                int(mid),
                str(team),
                players,
                [str(names.get(p, p)) for p in players],
                Data(x=torch.tensor(feats), edge_index=ei, edge_weight=ew),
                team_xg,
                float(opp["shot_xg"].fillna(0).sum()),
            )
        )
    return graphs


class _Encoder(nn.Module):
    def __init__(self, n_in: int, hidden: int, out: int, conv: str):
        super().__init__()
        if conv == "gat":
            self.c1 = GATConv(n_in, hidden // 4, heads=4, dropout=0.1)
            self.c2 = GATConv(hidden, out, heads=1, dropout=0.1)
        else:
            self.c1 = SAGEConv(n_in, hidden)
            self.c2 = SAGEConv(hidden, out)
        self.team_head = nn.Linear(out, 1)

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.c2(torch.relu(self.c1(x, edge_index)), edge_index)


class PlayerEmbeddingGNN:
    def __init__(
        self,
        hidden: int = 32,
        out: int = 16,
        epochs: int = 60,
        lr: float = 5e-3,
        seed: int = 42,
        conv: str = "sage",
        holdout_edge_fraction: float = 0.15,
    ):
        self.hidden, self.out, self.epochs, self.lr, self.seed, self.conv = hidden, out, epochs, lr, seed, conv
        self.holdout_edge_fraction = holdout_edge_fraction
        self._net: _Encoder | None = None
        self._mu: torch.Tensor | None = None
        self._sd: torch.Tensor | None = None
        self.embeddings_: pd.DataFrame = pd.DataFrame()
        self.link_auroc_: float | None = None

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        assert self._mu is not None and self._sd is not None
        return (x - self._mu) / self._sd

    @staticmethod
    def _split_edges(g: PassingGraph, frac: float, gen: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
        ei = g.data.edge_index
        n = ei.size(1)
        n_hold = max(1, int(n * frac)) if n >= 3 else 0
        perm = torch.randperm(n, generator=gen)
        return ei[:, perm[n_hold:]], ei[:, perm[:n_hold]]

    def fit(self, graphs: list[PassingGraph]) -> PlayerEmbeddingGNN:
        if not graphs:
            raise ValueError("no passing graphs to train on")
        torch.manual_seed(self.seed)
        gen = torch.Generator().manual_seed(self.seed)
        all_x = torch.cat([g.data.x for g in graphs])
        self._mu, self._sd = all_x.mean(0), all_x.std(0) + 1e-6
        self._net = _Encoder(len(NODE_FEATURES), self.hidden, self.out, self.conv)
        opt = torch.optim.Adam(self._net.parameters(), lr=self.lr)
        splits = [self._split_edges(g, self.holdout_edge_fraction, gen) for g in graphs]
        for _ in range(self.epochs):
            for g, (train_ei, _) in zip(graphs, splits, strict=True):
                opt.zero_grad()
                z = self._net.encode(self._norm(g.data.x), train_ei)
                src, dst = train_ei
                pos = (z[src] * z[dst]).sum(-1)
                neg = (z[src] * z[torch.randint(0, z.size(0), (src.size(0),), generator=gen)]).sum(-1)
                link_loss = nn.functional.binary_cross_entropy_with_logits(
                    torch.cat([pos, neg]), torch.cat([torch.ones_like(pos), torch.zeros_like(neg)])
                )
                share = torch.tensor(g.team_xg / (g.team_xg + g.opp_xg + 1e-6), dtype=torch.float32)
                team_loss = nn.functional.mse_loss(
                    torch.sigmoid(self._net.team_head(z.mean(0, keepdim=True))).squeeze(), share
                )
                (link_loss + team_loss).backward()
                opt.step()
        self.link_auroc_ = self._evaluate_links(graphs, splits, gen)
        self.embeddings_ = self.embed(graphs)
        return self

    def _evaluate_links(self, graphs: list[PassingGraph], splits, gen: torch.Generator) -> float | None:
        assert self._net is not None
        self._net.eval()
        auroc = torchmetrics.AUROC(task="binary")
        n_pairs = 0
        with torch.no_grad():
            for g, (train_ei, hold_ei) in zip(graphs, splits, strict=True):
                if hold_ei.size(1) == 0:
                    continue
                z = self._net.encode(self._norm(g.data.x), train_ei)
                src, dst = hold_ei
                pos = torch.sigmoid((z[src] * z[dst]).sum(-1))
                neg = torch.sigmoid((z[src] * z[torch.randint(0, z.size(0), (src.size(0),), generator=gen)]).sum(-1))
                auroc.update(torch.cat([pos, neg]), torch.cat([torch.ones_like(pos), torch.zeros_like(neg)]).long())
                n_pairs += 2 * src.size(0)
        return float(auroc.compute()) if n_pairs else None

    def embed(self, graphs: list[PassingGraph]) -> pd.DataFrame:
        if self._net is None:
            raise RuntimeError("fit() first")
        self._net.eval()
        rows = []
        with torch.no_grad():
            for g in graphs:
                z = self._net.encode(self._norm(g.data.x), g.data.edge_index).numpy()
                for pid, name, vec in zip(g.player_ids, g.player_names, z, strict=True):
                    rows.append(
                        {
                            "player_id": pid,
                            "player": name,
                            "team": g.team,
                            "match_id": g.match_id,
                            **{f"e{i}": float(v) for i, v in enumerate(vec)},
                        }
                    )
        df = pd.DataFrame(rows)
        if df.empty:
            return df
        emb_cols = [c for c in df.columns if c.startswith("e")]
        return (
            df.groupby(["player_id", "player"], as_index=False)
            .agg({**dict.fromkeys(emb_cols, "mean"), "team": "last", "match_id": "count"})
            .rename(columns={"match_id": "n_matches"})
        )

    def most_similar(self, player: str, k: int = 5) -> pd.DataFrame:
        df = self.embeddings_
        if df.empty:
            return df
        emb_cols = [c for c in df.columns if c.startswith("e")]
        target = df[df["player"].str.lower() == player.lower()]
        if target.empty:
            target = df[df["player"].str.lower().str.contains(player.lower(), regex=False)]
        if target.empty:
            return pd.DataFrame()
        v = target[emb_cols].to_numpy()[0]
        M = df[emb_cols].to_numpy()
        sim = M @ v / (np.linalg.norm(M, axis=1) * np.linalg.norm(v) + 1e-9)
        out = df.assign(similarity=sim).sort_values("similarity", ascending=False)
        return out[out["player"] != target["player"].iloc[0]].head(k)[["player", "team", "n_matches", "similarity"]]

    def card(self) -> dict:
        return {
            "name": "player_embedding_gnn",
            "family": f"{'GAT' if self.conv == 'gat' else 'GraphSAGE'} (PyTorch Geometric), self-supervised link prediction + team xG-share head",
            "node_features": NODE_FEATURES,
            "embedding_dim": self.out,
            "epochs": self.epochs,
            "held_out_edge_fraction": self.holdout_edge_fraction,
            "link_prediction_auroc": self.link_auroc_,
            "data": "StatsBomb Open Data events (attribution required)",
            "use": "similarity search, team-strength descriptors",
        }
