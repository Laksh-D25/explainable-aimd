"""Datasets.

One dataset class serves training and evaluation, because the corrected
architecture makes a song -- not a clip -- the unit of prediction: `song_pool`
is inside the model, so a batch item is always a song's worth of clips.

Two disciplines are enforced here rather than left to the caller:

* **Augmentation is train-only.** Passing an augmenter with a non-train split
  raises. Augmenting validation or test would leak the training distribution
  into the numbers meant to measure generalisation.
* **Augmentation and robustness perturbation are mutually exclusive.** Stacking
  a seen training condition under a held-out eval condition would make the
  robustness table unattributable to either.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .audio import fit_length, load_audio, pad_clips, rms_normalize, segment
from .perturb import Perturbation, TrainAugment


@dataclass
class ClipSpec:
    clip_seconds: float = 10.0
    clips_per_song: int = 8
    max_clips: int = 12
    sample_rate: int = 24_000


class SongClipsDataset(Dataset):
    """Yields one song per item as a padded, masked stack of clips.

    Args:
        manifest: rows for a single split, with the canonical manifest columns.
        spec: clip geometry.
        split: which split these rows are; gates augmentation.
        augment: train-time augmentation, train split only.
        perturbation: a single held-out condition for the robustness protocol.
        taxonomy_classes: index order for the auxiliary head's labels.
        sources: index order for closed-set attribution, or None to omit.
    """

    def __init__(
        self,
        manifest: pd.DataFrame,
        spec: ClipSpec | None = None,
        split: str = "train",
        augment: TrainAugment | None = None,
        perturbation: Perturbation | None = None,
        taxonomy_classes: tuple[str, ...] = ("real", "full_fake", "half_fake", "mostly_fake"),
        sources: tuple[str, ...] | None = None,
        seed: int = 0,
    ):
        if augment is not None and split != "train":
            raise ValueError(
                f"augmentation requested for split={split!r}. Training augmentation must "
                "not touch val/test -- it would leak the training distribution into the "
                "numbers meant to measure generalisation."
            )
        if augment is not None and perturbation is not None:
            raise ValueError(
                "augment and perturbation are mutually exclusive: stacking a seen "
                "training condition under a held-out eval condition makes the robustness "
                "result attributable to neither."
            )

        self.manifest = manifest.reset_index(drop=True)
        self.spec = spec or ClipSpec()
        self.split = split
        self.augment = augment
        self.perturbation = perturbation
        self.taxonomy_classes = taxonomy_classes
        self.sources = sources
        self.seed = seed
        # Train draws random offsets so a song is seen differently each epoch;
        # eval is uniform and deterministic so a score belongs to the song.
        self.strategy = "random" if split == "train" else "uniform"

    def __len__(self) -> int:
        return len(self.manifest)

    def _rng(self, index: int) -> np.random.Generator:
        # Per-item seeding keeps workers independent and the run reproducible.
        return np.random.default_rng((self.seed, index))

    def __getitem__(self, index: int) -> dict:
        row = self.manifest.iloc[index]
        spec = self.spec
        rng = self._rng(index)

        # RMS rather than peak: peak normalisation leaves an 0.81-AUROC
        # loudness shortcut between mastered and generated audio.
        wav = rms_normalize(load_audio(row["path"], spec.sample_rate))
        clips = segment(
            wav,
            clip_seconds=spec.clip_seconds,
            n_clips=spec.clips_per_song,
            sample_rate=spec.sample_rate,
            strategy=self.strategy,
            rng=rng,
        )

        # Perturbations may not preserve length (MP3 adds encoder padding), so
        # every clip is pinned back to the configured size.
        clip_samples = clips.shape[1]
        if self.augment is not None:
            clips = np.stack(
                [fit_length(self.augment(c, spec.sample_rate), clip_samples) for c in clips]
            )
        elif self.perturbation is not None:
            clips = np.stack(
                [
                    fit_length(
                        np.asarray(self.perturbation(c, spec.sample_rate), dtype=np.float32),
                        clip_samples,
                    )
                    for c in clips
                ]
            )

        padded, mask = pad_clips(clips, spec.max_clips)

        item = {
            "clips": torch.from_numpy(padded),
            "clip_mask": torch.from_numpy(mask),
            "label": torch.tensor(float(row["label"])),
            "taxonomy": torch.tensor(self.taxonomy_classes.index(row["taxonomy"])),
            "song_id": str(row["song_id"]),
        }
        if self.sources is not None:
            source = row.get("source")
            item["source"] = torch.tensor(
                self.sources.index(source) if source in self.sources else -1
            )
        return item


def collate(batch: list[dict]) -> dict:
    """Stack songs into [B, C, N] plus the clip mask the song pool needs."""
    out = {
        "clips": torch.stack([b["clips"] for b in batch]),
        "clip_mask": torch.stack([b["clip_mask"] for b in batch]),
        "label": torch.stack([b["label"] for b in batch]),
        "taxonomy": torch.stack([b["taxonomy"] for b in batch]),
        "song_id": [b["song_id"] for b in batch],
    }
    if "source" in batch[0]:
        out["source"] = torch.stack([b["source"] for b in batch])
    return out
