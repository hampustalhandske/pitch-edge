"""Sequence models over each team's recent match history — PyTorch Lightning.

Two encoders share one training recipe:
* `gru`         — GRU over the last `seq_len` team-centric step vectors (the deep-learning entry).
* `transformer` — TransformerEncoder with learned positional embeddings and mean pooling.

Recipe (the parts that make a deep model defensible in a walk-forward backtest):
* time-ordered validation split inside each training fold (last 15 % by date — never random),
* early stopping on validation log-loss, ReduceLROnPlateau, gradient clipping, AdamW + weight decay,
* temperature scaling fitted on the same validation slice (calibration before staking),
* torchmetrics for validation log-loss / accuracy; deterministic seeding.
Sequences use matches strictly before the fixture date; `observe()` only ever adds realised results.
"""

from __future__ import annotations

import logging
import warnings

import lightning as L
import numpy as np
import pandas as pd
import torch
import torchmetrics
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from pitch_edge.models.base import MatchModel, to_frame

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", message=".*does not have many workers.*")
warnings.filterwarnings("ignore", message=".*GPU available.*")

STEP_FEATURES = ["gf", "ga", "sf", "sa", "stf", "sta", "is_home", "pts", "days_gap"]


# ----------------------------------------------------------------------------- networks
class _GRUEncoder(nn.Module):
    def __init__(self, n_in: int, hidden: int):
        super().__init__()
        self.gru = nn.GRU(n_in, hidden, batch_first=True)
        self.out_dim = hidden

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, h = self.gru(x)
        return h[-1]


class _TransformerEncoder(nn.Module):
    def __init__(self, n_in: int, hidden: int, seq_len: int, n_heads: int = 4, n_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Linear(n_in, hidden)
        self.pos = nn.Parameter(torch.zeros(1, seq_len, hidden))
        layer = nn.TransformerEncoderLayer(
            hidden, n_heads, dim_feedforward=hidden * 2, dropout=dropout, batch_first=True
        )
        self.enc = nn.TransformerEncoder(layer, n_layers)
        self.out_dim = hidden

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.enc(self.proj(x) + self.pos[:, : x.size(1)])
        return h.mean(dim=1)


class SequenceLitModule(L.LightningModule):
    def __init__(
        self, n_in: int, seq_len: int, hidden: int = 32, encoder: str = "gru", n_extra: int = 2, lr: float = 2e-3
    ):
        super().__init__()
        self.save_hyperparameters()
        self.encoder = _GRUEncoder(n_in, hidden) if encoder == "gru" else _TransformerEncoder(n_in, hidden, seq_len)
        self.head = nn.Sequential(
            nn.Linear(2 * self.encoder.out_dim + n_extra, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, 3)
        )
        self.loss_fn = nn.CrossEntropyLoss()
        self.val_ll = torchmetrics.MeanMetric()
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=3)
        self.temperature = nn.Parameter(torch.ones(1), requires_grad=False)

    def forward(self, home: torch.Tensor, away: torch.Tensor, extra: torch.Tensor) -> torch.Tensor:
        return self.head(torch.cat([self.encoder(home), self.encoder(away), extra], dim=1))

    def training_step(self, batch, _):
        h, a, e, y = batch
        loss = self.loss_fn(self(h, a, e), y)
        self.log("train_loss", loss, prog_bar=False)
        return loss

    def validation_step(self, batch, _):
        h, a, e, y = batch
        logits = self(h, a, e)
        loss = self.loss_fn(logits, y)
        self.val_ll.update(loss, weight=len(y))
        self.val_acc.update(logits.argmax(dim=1), y)
        self.log("val_loss", loss, prog_bar=False)

    def on_validation_epoch_end(self):
        self.log("val_log_loss", self.val_ll.compute())
        self.log("val_accuracy", self.val_acc.compute())
        self.val_ll.reset()
        self.val_acc.reset()

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=2)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "monitor": "val_loss"}}

    def fit_temperature(self, logits: torch.Tensor, y: torch.Tensor, min_rows: int = 300) -> float:
        """Temperature scaling (Guo et al. 2017) on the validation slice.

        Only softening (T >= 1) is allowed and only with enough rows: on tiny slices a sharpening
        temperature fits noise and produces over-confident, worse-than-uniform test log-loss."""
        if len(y) < min_rows:
            self.temperature.data = torch.tensor([1.0])
            return 1.0
        temps = torch.linspace(1.0, 2.5, 31)
        losses = [nn.functional.cross_entropy(logits / t, y).item() for t in temps]
        best = float(temps[int(np.argmin(losses))])
        self.temperature.data = torch.tensor([best])
        return best


# ----------------------------------------------------------------------------- MatchModel
class SequenceMatchModel(MatchModel):
    """Shared MatchModel wrapper; `GRUSequenceModel` / `TransformerSequenceModel` pick the encoder."""

    encoder = "gru"
    name = "gru_sequence"

    def __init__(
        self,
        seq_len: int = 10,
        hidden: int = 32,
        epochs: int = 25,
        lr: float = 5e-4,
        batch_size: int = 256,
        seed: int = 42,
        patience: int = 6,
        val_fraction: float = 0.15,
        accelerator: str = "cpu",
    ):
        self.seq_len, self.hidden, self.epochs, self.lr, self.batch_size = seq_len, hidden, epochs, lr, batch_size
        self.seed, self.patience, self.val_fraction, self.accelerator = seed, patience, val_fraction, accelerator
        self._net: SequenceLitModule | None = None
        self._history = pd.DataFrame()
        self._by_team: dict[str, pd.DataFrame] = {}
        self._mu: np.ndarray | None = None
        self._sd: np.ndarray | None = None
        self.last_fit_: dict = {}

    # -------------------------------------------------------------- history
    @staticmethod
    def _to_long(df: pd.DataFrame) -> pd.DataFrame:
        def side(p: str, o: str, is_home: int) -> pd.DataFrame:
            g = lambda c: df.get(c, pd.Series(np.nan, index=df.index)).to_numpy(dtype=float)  # noqa: E731
            return pd.DataFrame(
                {
                    "team": df[f"{p}_team"].to_numpy(),
                    "date": pd.to_datetime(df["date"]).to_numpy(),
                    "gf": g(f"{p}_goals"),
                    "ga": g(f"{o}_goals"),
                    "sf": g(f"{p}_shots"),
                    "sa": g(f"{o}_shots"),
                    "stf": g(f"{p}_shots_on_target"),
                    "sta": g(f"{o}_shots_on_target"),
                    "is_home": float(is_home),
                }
            )

        long = pd.concat([side("home", "away", 1), side("away", "home", 0)], ignore_index=True)
        long["pts"] = np.where(long["gf"] > long["ga"], 3.0, np.where(long["gf"] == long["ga"], 1.0, 0.0))
        return long.sort_values(["team", "date"]).reset_index(drop=True)

    def observe(self, realized: pd.DataFrame) -> None:
        self._history = pd.concat([self._history, self._to_long(realized)], ignore_index=True)
        self._history = self._history.drop_duplicates().sort_values(["team", "date"]).reset_index(drop=True)
        self._by_team = {t: g for t, g in self._history.groupby("team", sort=False)}  # noqa: C416

    def _sequence(self, team: str, before: pd.Timestamp) -> np.ndarray:
        seq = np.zeros((self.seq_len, len(STEP_FEATURES)), dtype=np.float32)
        g = self._by_team.get(team)
        if g is None:
            return seq
        prior = g[g["date"] < before].tail(self.seq_len)
        if prior.empty:
            return seq
        gaps = (
            np.diff(np.concatenate([prior["date"].to_numpy(), [np.datetime64(before)]]))
            .astype("timedelta64[D]")
            .astype(float)
        )
        vals = np.nan_to_num(prior[STEP_FEATURES[:-1]].to_numpy(dtype=np.float32), nan=0.0)
        block = np.concatenate([vals, np.clip(gaps, 0, 60)[:, None].astype(np.float32)], axis=1)
        seq[-len(block) :] = block
        return seq

    def _tensors(self, X: pd.DataFrame) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dates = pd.to_datetime(X["date"])
        home = np.stack([self._sequence(t, d) for t, d in zip(X["home_team"], dates, strict=True)])
        away = np.stack([self._sequence(t, d) for t, d in zip(X["away_team"], dates, strict=True)])
        if self._mu is not None and self._sd is not None:
            home = (home - self._mu) / self._sd
            away = (away - self._mu) / self._sd
        elo_diff = X.get("elo_diff", pd.Series(0.0, index=X.index)).fillna(0).to_numpy(dtype=np.float32) / 400.0
        extra = np.stack([elo_diff, np.ones_like(elo_diff)], axis=1)
        t = lambda a: torch.tensor(np.asarray(a, dtype=np.float32))  # noqa: E731
        return t(home), t(away), t(extra)

    # ------------------------------------------------------------------ fit
    def fit(self, train: pd.DataFrame) -> SequenceMatchModel:
        L.seed_everything(self.seed, workers=True, verbose=False)
        self._history = pd.DataFrame()
        self.observe(train)
        raw = np.nan_to_num(self._history[STEP_FEATURES[:-1]].to_numpy(dtype=np.float32), nan=0.0)
        self._mu = np.concatenate([raw.mean(axis=0), [7.0]]).astype(np.float32)
        self._sd = np.concatenate([raw.std(axis=0) + 1e-6, [7.0]]).astype(np.float32)

        train = train.sort_values("date")
        n_val = max(int(len(train) * self.val_fraction), 64) if len(train) > 400 else 0
        tr, va = (train.iloc[:-n_val], train.iloc[-n_val:]) if n_val else (train, train.tail(min(64, len(train))))
        h, a, e = self._tensors(tr)
        y = torch.tensor(tr["result"].to_numpy(dtype=np.int64))
        hv, av, ev = self._tensors(va)
        yv = torch.tensor(va["result"].to_numpy(dtype=np.int64))

        self._net = SequenceLitModule(len(STEP_FEATURES), self.seq_len, self.hidden, self.encoder, lr=self.lr)
        trainer = L.Trainer(
            max_epochs=self.epochs,
            accelerator=self.accelerator,
            devices=1,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
            gradient_clip_val=1.0,
            deterministic=True,
            # early stopping only when the validation slice is large enough to be a signal, not noise
            callbacks=(
                [L.pytorch.callbacks.EarlyStopping(monitor="val_loss", patience=self.patience, mode="min")]
                if len(va) >= 200
                else []
            ),
        )
        trainer.fit(
            self._net,
            DataLoader(TensorDataset(h, a, e, y), batch_size=self.batch_size, shuffle=True),
            DataLoader(TensorDataset(hv, av, ev, yv), batch_size=1024),
        )
        self._net.eval()
        with torch.no_grad():
            temp = self._net.fit_temperature(self._net(hv, av, ev), yv)
        self.last_fit_ = {
            "epochs_run": trainer.current_epoch,
            "val_log_loss": float(trainer.callback_metrics.get("val_log_loss", np.nan)),
            "val_accuracy": float(trainer.callback_metrics.get("val_accuracy", np.nan)),
            "temperature": temp,
            "n_train": len(tr),
            "n_val": len(va),
        }
        return self

    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        if self._net is None:
            raise RuntimeError("fit() first")
        self._net.eval()
        with torch.no_grad():
            h, a, e = self._tensors(X)
            probs = torch.softmax(self._net(h, a, e) / self._net.temperature, dim=1).cpu().numpy()
        return to_frame(probs, X.index)

    def card(self) -> dict:
        return {
            **super().card(),
            "family": f"{self.encoder} sequence encoder over per-team match history (PyTorch Lightning)",
            "seq_len": self.seq_len,
            "hidden": self.hidden,
            "max_epochs": self.epochs,
            "lr": self.lr,
            "patience": self.patience,
            "step_features": STEP_FEATURES,
            "extra_features": ["elo_diff/400"],
            "training": "time-ordered 85/15 split inside each fold, early stopping on val loss, ReduceLROnPlateau, grad clip 1.0, AdamW",
            "calibration": "temperature scaling on the validation slice, then isotonic in the backtester",
            "last_fit": self.last_fit_,
            "leakage_checks": [
                "sequences use matches strictly before fixture date",
                "validation slice is the most recent 15 %, never shuffled",
                "observe() only adds realised results",
            ],
        }


class GRUSequenceModel(SequenceMatchModel):
    encoder = "gru"
    name = "gru_sequence"


class TransformerSequenceModel(SequenceMatchModel):
    encoder = "transformer"
    name = "transformer_sequence"
