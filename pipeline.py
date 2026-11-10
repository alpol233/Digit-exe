from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from digitsynth import synth
from networks import build


@dataclass
class Hyper:
    arch: str = "small_conv"
    epochs: int = 20
    batch: int = 128
    lr: float = 1e-3
    per_class: int = 300
    val_share: float = 0.2
    patience: int = 5
    weight_decay: float = 0.0
    extra: dict = field(default_factory=dict)


class Trainer:
    def __init__(self, hyper: Hyper, weights: Path = Path("checkpoint.pt"), device=None):
        self.hyper = hyper
        self.weights = weights
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = build(hyper.arch).to(self.device)
        self.loss_fn = nn.CrossEntropyLoss()
        self.opt = torch.optim.Adam(
            self.net.parameters(), lr=hyper.lr, weight_decay=hyper.weight_decay
        )
        self.sched = torch.optim.lr_scheduler.StepLR(self.opt, step_size=8, gamma=0.5)

    def _split(self):
        X, y = synth(self.hyper.per_class)
        rng = np.random.default_rng(0)
        order = rng.permutation(len(X))
        cut = int((1.0 - self.hyper.val_share) * len(X))
        tr, va = order[:cut], order[cut:]
        return (
            TensorDataset(torch.from_numpy(X[tr]), torch.from_numpy(y[tr])),
            TensorDataset(torch.from_numpy(X[va]), torch.from_numpy(y[va])),
        )

    def _epoch(self, loader, train):
        self.net.train(train)
        total_loss = total_correct = total = 0
        ctx = torch.enable_grad() if train else torch.no_grad()
        with ctx:
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                if train:
                    self.opt.zero_grad()
                logits = self.net(xb)
                loss = self.loss_fn(logits, yb)
                if train:
                    loss.backward()
                    self.opt.step()
                total_loss += loss.item() * xb.size(0)
                total_correct += (logits.argmax(1) == yb).sum().item()
                total += xb.size(0)
        return total_loss / total, total_correct / total

    def fit(self):
        tr_ds, va_ds = self._split()
        tr_dl = DataLoader(tr_ds, batch_size=self.hyper.batch, shuffle=True)
        va_dl = DataLoader(va_ds, batch_size=self.hyper.batch)

        best = 0.0
        stale = 0
        history = []
        for ep in range(1, self.hyper.epochs + 1):
            tr_loss, tr_acc = self._epoch(tr_dl, True)
            self.sched.step()
            _, va_acc = self._epoch(va_dl, False)
            history.append({"epoch": ep, "loss": tr_loss, "train": tr_acc, "val": va_acc})
            print(
                f"ep {ep:02d}/{self.hyper.epochs}  loss={tr_loss:.4f}  "
                f"train={tr_acc * 100:.2f}%  val={va_acc * 100:.2f}%"
            )
            if va_acc > best:
                best, stale = va_acc, 0
                self._save(best, ep)
                print("  saved:", self.weights)
            else:
                stale += 1
                if stale >= self.hyper.patience:
                    print("  early stop")
                    break

        print(f"best val: {best * 100:.2f}%  file={self.weights.resolve()}")
        return {"best_val": best, "history": history}

    def _save(self, best_acc, epoch):
        payload = {
            "arch": self.hyper.arch,
            "state": self.net.state_dict(),
            "best_val": best_acc,
            "epoch": epoch,
            "hyper": self.hyper.__dict__,
        }
        torch.save(payload, self.weights)

    @staticmethod
    def load(weights: Path = Path("checkpoint.pt"), device=None):
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        payload = torch.load(weights, map_location=device)
        net = build(payload.get("arch", "small_conv")).to(device)
        net.load_state_dict(payload["state"])
        net.eval()
        return net, device, payload