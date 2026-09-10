"""Training and inference loops.

Only the ~0.6 M-parameter tail trains; MERT's 94 M parameters are frozen. That
shapes two choices here:

* The backbone runs under `no_grad` on the training path, so activations inside
  its 12 transformer layers are never retained. This is what makes training
  fit on a 4 GB laptop GPU at all.
* Mixed precision applies to the backbone forward pass, where the memory goes.
  The trainable head stays in fp32 -- it is small enough that fp16 buys nothing
  and costs numerical headroom.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..models.detector import Detector
from .losses import DetectorLoss


@dataclass
class EpochStats:
    loss: float
    parts: dict[str, float] = field(default_factory=dict)
    n_songs: int = 0


def _to_device(batch: dict, device: str) -> dict:
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


def run_epoch(
    model: Detector,
    loader: DataLoader,
    loss_fn: DetectorLoss,
    optimizer: torch.optim.Optimizer | None = None,
    device: str = "cpu",
    amp: bool = False,
    scaler: torch.amp.GradScaler | None = None,
    grad_clip: float | None = 1.0,
) -> EpochStats:
    """One pass. Pass `optimizer=None` to evaluate."""
    training = optimizer is not None
    model.train(training)

    totals: dict[str, float] = {}
    total_loss, n = 0.0, 0

    for batch in loader:
        batch = _to_device(batch, device)
        with torch.amp.autocast(device_type=device.split(":")[0], enabled=amp):
            out = model.forward(
                batch["clips"], clip_mask=batch["clip_mask"], backbone_no_grad=True
            )
        # Compute the loss in fp32: autocast leaves logits in fp16, and BCE on
        # fp16 logits loses resolution exactly where the decision is made.
        out.binary_logit = out.binary_logit.float()
        if out.taxonomy_logits is not None:
            out.taxonomy_logits = out.taxonomy_logits.float()
        loss, parts = loss_fn(out, batch)

        if training:
            optimizer.zero_grad(set_to_none=True)
            if scaler is not None:
                scaler.scale(loss).backward()
                if grad_clip:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if grad_clip:
                    torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), grad_clip)
                optimizer.step()

        size = batch["label"].shape[0]
        total_loss += loss.item() * size
        for k, v in parts.items():
            totals[k] = totals.get(k, 0.0) + v * size
        n += size

    return EpochStats(
        loss=total_loss / max(n, 1),
        parts={k: v / max(n, 1) for k, v in totals.items()},
        n_songs=n,
    )


@torch.no_grad()
def predict(
    model: Detector, loader: DataLoader, device: str = "cpu", amp: bool = False
) -> dict[str, np.ndarray]:
    """Song-level predictions. Returns logits as well as probabilities so that
    temperature scaling can be fitted afterwards without a second pass."""
    model.eval()
    logits, labels, taxonomy, song_ids = [], [], [], []

    for batch in loader:
        batch = _to_device(batch, device)
        with torch.amp.autocast(device_type=device.split(":")[0], enabled=amp):
            out = model.forward(
                batch["clips"], clip_mask=batch["clip_mask"], backbone_no_grad=True
            )
        logits.append(out.binary_logit.float().cpu())
        labels.append(batch["label"].cpu())
        taxonomy.append(batch["taxonomy"].cpu())
        song_ids.extend(batch["song_id"])

    logit = torch.cat(logits).numpy()
    return {
        "logit": logit,
        "prob": 1 / (1 + np.exp(-logit)),
        "label": torch.cat(labels).numpy().astype(int),
        "taxonomy": torch.cat(taxonomy).numpy().astype(int),
        "song_id": np.array(song_ids),
    }
