"""Orchestration shared by the CLI and the Kaggle notebook.

Keeping training and evaluation here rather than in either entry point means
the notebook that produces the reportable numbers runs exactly the same code
as the local CLI. A notebook that quietly diverges from the repo is how
irreproducible results happen.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .calibrate.temperature import TemperatureScaler
from .data.datasets import ClipSpec, SongClipsDataset, collate
from .data.perturb import TrainAugment
from .eval.metrics import summary
from .eval.protocols import build_loader, rows_for_split
from .models.backbone import MertBackbone
from .models.detector import Detector
from .train.losses import DetectorLoss, class_weights
from .train.loop import predict, run_epoch


@dataclass
class TrainConfig:
    epochs: int = 15
    batch_size: int = 8
    lr: float = 1e-3
    weight_decay: float = 0.01
    aux_loss_weight: float = 0.3
    clip_seconds: float = 10.0
    clips_per_song: int = 8
    #: None means "exactly clips_per_song". Segmentation only returns fewer
    #: than clips_per_song for songs shorter than one clip, so a larger value
    #: just pads -- and padding costs a full MERT forward per padded clip.
    max_clips: int | None = None
    augment: bool = True
    augment_probability: float = 0.5
    num_workers: int = 2
    amp: bool = True
    seed: int = 1337
    limit_songs: int | None = None

    def spec(self) -> ClipSpec:
        return ClipSpec(
            clip_seconds=self.clip_seconds,
            clips_per_song=self.clips_per_song,
            max_clips=self.max_clips or self.clips_per_song,
        )


#: Leave this much VRAM for the desktop. The RTX 3050 drives both the display
#: and the workload; training at 3.0 GB of 3.75 GB starved the compositor and
#: froze the session hard enough to need `systemctl restart lightdm`.
DISPLAY_VRAM_RESERVE_GB = 1.3


def resolve_device(requested: str = "auto") -> str:
    if requested != "auto":
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


def cap_vram(device: str, reserve_gb: float = DISPLAY_VRAM_RESERVE_GB) -> None:
    """Bound this process's VRAM so the display server keeps working.

    Without a cap PyTorch will happily grow its cache until the compositor
    cannot allocate, at which point the desktop locks up rather than raising an
    error anyone can see. A hard fraction turns that silent freeze into an
    ordinary OutOfMemoryError, which is recoverable and debuggable.
    """
    if not device.startswith("cuda"):
        return
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    fraction = max(0.35, min(0.85, (total - reserve_gb) / total))
    torch.cuda.set_per_process_memory_fraction(fraction)
    print(f"  vram capped at {fraction:.0%} of {total:.2f} GB "
          f"(~{reserve_gb:.1f} GB reserved for the display)", flush=True)


def build_model(device: str, model_name: str = "m-a-p/MERT-v1-95M", **kwargs) -> Detector:
    cap_vram(device)
    return Detector(backbone=MertBackbone(model_name=model_name), **kwargs).to(device)


def _taxonomy_weights(manifest: pd.DataFrame, classes: tuple[str, ...]) -> torch.Tensor:
    counts = torch.tensor([int((manifest["taxonomy"] == c).sum()) for c in classes])
    return class_weights(counts)


def train_model(
    model: Detector,
    manifest: pd.DataFrame,
    config: TrainConfig,
    device: str = "cpu",
    checkpoint_dir: Path | None = None,
    log=print,
) -> dict:
    """Train the head, selecting the checkpoint by validation F1.

    Selection is on F1 rather than loss because the deployment question is how
    well the detector separates classes, not how confident it is -- confidence
    is handled afterwards by calibration.
    """
    torch.manual_seed(config.seed)
    spec = config.spec()
    from .models.heads import TAXONOMY_CLASSES

    classes = TAXONOMY_CLASSES if model.taxonomy_head is not None else ()

    train_rows = rows_for_split(manifest, "train")
    if config.limit_songs:
        train_rows = train_rows.head(config.limit_songs)

    augment = TrainAugment(config.augment_probability, seed=config.seed) if config.augment else None
    train_ds = SongClipsDataset(train_rows, spec, split="train", augment=augment, seed=config.seed)
    train_dl = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate,
        num_workers=config.num_workers,
        drop_last=False,
    )
    val_dl = build_loader(manifest, spec, "val", batch_size=config.batch_size,
                          num_workers=config.num_workers)

    loss_fn = DetectorLoss(
        aux_weight=config.aux_loss_weight,
        taxonomy_weights=_taxonomy_weights(train_rows, classes).to(device) if classes else None,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.trainable_parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
    amp = config.amp and device.startswith("cuda")
    scaler = torch.amp.GradScaler(device) if amp else None

    history, best = [], {"f1": -1.0, "epoch": -1}
    for epoch in range(config.epochs):
        stats = run_epoch(model, train_dl, loss_fn, optimizer, device, amp, scaler)
        val = predict(model, val_dl, device=device, amp=amp)
        metrics = summary(val["label"], val["prob"])
        scheduler.step()

        history.append({"epoch": epoch, "train_loss": stats.loss, **metrics})
        log(
            f"epoch {epoch + 1}/{config.epochs}  loss {stats.loss:.4f}  "
            f"val f1 {metrics['f1']:.4f}  auroc {metrics['auroc']:.4f}"
        )

        if metrics["f1"] > best["f1"]:
            best = {"f1": metrics["f1"], "epoch": epoch}
            if checkpoint_dir:
                checkpoint_dir = Path(checkpoint_dir)
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                # Only the trainable tail is saved: the backbone is frozen and
                # reloadable from its checkpoint name, so storing it would waste
                # ~380 MB per checkpoint for no information.
                torch.save(
                    {
                        "state_dict": {
                            k: v for k, v in model.state_dict().items()
                            if not k.startswith("backbone.")
                        },
                        "config": asdict(config),
                        "epoch": epoch,
                        "val_f1": metrics["f1"],
                    },
                    checkpoint_dir / "best.pt",
                )

    return {"history": pd.DataFrame(history), "best": best}


def load_checkpoint(model: Detector, path: Path, device: str = "cpu") -> dict:
    """Restore the trainable tail; the frozen backbone comes from its own weights."""
    payload = torch.load(path, map_location=device, weights_only=False)
    missing, unexpected = model.load_state_dict(payload["state_dict"], strict=False)
    unexpected = [k for k in unexpected if not k.startswith("backbone.")]
    if unexpected:
        raise ValueError(f"unexpected keys in checkpoint: {unexpected[:5]}")
    if any(not k.startswith("backbone.") for k in missing):
        raise ValueError(
            f"checkpoint is missing trainable weights: "
            f"{[k for k in missing if not k.startswith('backbone.')][:5]}"
        )
    return payload


def calibrate_and_threshold(
    model: Detector,
    manifest: pd.DataFrame,
    spec: ClipSpec,
    target_fpr: float = 0.01,
    device: str = "cpu",
    amp: bool = False,
) -> tuple[TemperatureScaler, float, dict]:
    """Fit temperature on validation and report the ECE improvement."""
    from .calibrate.temperature import threshold_for_target_fpr

    out = predict(model, build_loader(manifest, spec, "val"), device=device, amp=amp)
    before = summary(out["label"], out["prob"])
    scaler = TemperatureScaler().fit(out["logit"], out["label"])
    after = summary(out["label"], scaler.transform(out["logit"]))
    threshold = threshold_for_target_fpr(out["logit"], out["label"], target_fpr, scaler)
    return scaler, threshold, {
        "temperature": scaler.temperature,
        "ece_before": before["ece"],
        "ece_after": after["ece"],
        "threshold": threshold,
    }
