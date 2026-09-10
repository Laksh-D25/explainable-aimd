"""Training objective.

The binary real/fake term is primary -- it is the number compared against the
base paper's F1 0.99 -> 0.629 collapse, and the only head that transfers to
FakeMusicCaps. The 4-way SONICS taxonomy is auxiliary: useful as a training
signal about *how* a track was generated, but down-weighted so it cannot
dominate the objective that actually gets reported.

Both terms need class weighting for different reasons. The binary split is
roughly balanced in SONICS (~48k real vs ~49k fake) but a subset may not be,
and the taxonomy is unbalanced by construction -- the three fake subtypes split
one half of the data between them.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..models.detector import DetectorOutput


def class_weights(counts: torch.Tensor, clamp: float = 10.0) -> torch.Tensor:
    """Inverse-frequency weights, normalised to mean 1 and clamped.

    Clamping matters: an unclamped inverse frequency on a class with a handful
    of examples produces a weight large enough to destabilise training, which
    looks like a learning-rate problem rather than a weighting one.
    """
    counts = counts.float().clamp(min=1.0)
    w = counts.sum() / (len(counts) * counts)
    return (w / w.mean()).clamp(max=clamp)


class DetectorLoss(nn.Module):
    def __init__(
        self,
        aux_weight: float = 0.3,
        source_weight: float = 0.0,
        pos_weight: torch.Tensor | None = None,
        taxonomy_weights: torch.Tensor | None = None,
    ):
        super().__init__()
        self.aux_weight = aux_weight
        self.source_weight = source_weight
        self.register_buffer("pos_weight", pos_weight)
        self.register_buffer("taxonomy_weights", taxonomy_weights)

    def forward(self, out: DetectorOutput, batch: dict) -> tuple[torch.Tensor, dict[str, float]]:
        binary = F.binary_cross_entropy_with_logits(
            out.binary_logit, batch["label"].float(), pos_weight=self.pos_weight
        )
        total = binary
        parts = {"binary": binary.item()}

        if out.taxonomy_logits is not None and self.aux_weight > 0:
            aux = F.cross_entropy(
                out.taxonomy_logits, batch["taxonomy"], weight=self.taxonomy_weights
            )
            total = total + self.aux_weight * aux
            parts["taxonomy"] = aux.item()

        if out.source_logits is not None and self.source_weight > 0 and "source" in batch:
            # -1 marks a row whose generator is unknown (e.g. a real song), which
            # must not contribute to an attribution loss.
            src = F.cross_entropy(out.source_logits, batch["source"], ignore_index=-1)
            if torch.isfinite(src):
                total = total + self.source_weight * src
                parts["source"] = src.item()

        parts["total"] = total.item()
        return total, parts
