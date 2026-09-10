"""Explanation faithfulness.

The literature review notes that explanation fidelity remains an open metric,
and it is the gap that makes a three-level XAI layer risky: three plausible
pictures that do not reflect the model are worse than one honest number,
because they invite confident action on a wrong reason.

So every explanation this project produces is scored the same way. Mask the
regions an explanation calls important and watch the logit move:

* **deletion** -- remove the top-k regions. A faithful explanation causes a
  large drop.
* **insertion** -- keep only the top-k regions. A faithful explanation recovers
  the prediction quickly.
* **random baseline** -- the same operation on randomly chosen regions. This is
  the comparison that matters: a deletion curve that looks steep means nothing
  until it is steeper than chance, because masking any audio degrades a
  detector somewhat.

Report the gap against the random baseline, not the raw curve.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ..models.detector import Detector


@dataclass
class FaithfulnessCurve:
    fractions: np.ndarray
    scores: np.ndarray
    random_scores: np.ndarray

    @property
    def auc(self) -> float:
        return float(np.trapezoid(self.scores, self.fractions))

    @property
    def random_auc(self) -> float:
        return float(np.trapezoid(self.random_scores, self.fractions))

    @property
    def gap(self) -> float:
        """Area between the explanation curve and chance.

        For deletion this is negative when the explanation is faithful (the
        logit falls faster than chance); for insertion, positive.
        """
        return self.auc - self.random_auc


def _mask_frames(
    wav: torch.Tensor, order: np.ndarray, n_masked: int, samples_per_frame: int, keep: bool
) -> torch.Tensor:
    """Zero the selected frames (or everything but them, when `keep`)."""
    out = wav.clone()
    chosen = order[:n_masked]
    if keep:
        masked = torch.zeros_like(out)
        for frame in chosen:
            lo = int(frame) * samples_per_frame
            masked[..., lo : lo + samples_per_frame] = out[..., lo : lo + samples_per_frame]
        return masked
    for frame in chosen:
        lo = int(frame) * samples_per_frame
        out[..., lo : lo + samples_per_frame] = 0.0
    return out


@torch.no_grad()
def _logit(model: Detector, wav: torch.Tensor, clip_mask: torch.Tensor | None) -> float:
    return float(model.forward(wav, clip_mask=clip_mask).binary_logit.item())


def deletion_curve(
    model: Detector,
    wav: torch.Tensor,
    relevance: np.ndarray,
    clip_mask: torch.Tensor | None = None,
    steps: int = 10,
    seed: int = 0,
    keep: bool = False,
) -> FaithfulnessCurve:
    """Progressively mask the most-relevant frames and track the logit.

    Args:
        wav: [1, C, N] one song.
        relevance: [C, T] per-clip relevance from an explanation method.
        keep: False for deletion, True for insertion.
    """
    if wav.dim() != 3 or wav.shape[0] != 1:
        raise ValueError(f"expected one song as [1, C, N], got {tuple(wav.shape)}")

    n_clips, n_frames = relevance.shape
    samples_per_frame = wav.shape[-1] // n_frames
    flat = relevance.ravel()
    order = np.argsort(flat)[::-1]
    rng = np.random.default_rng(seed)
    random_order = rng.permutation(len(flat))

    def evaluate(sequence: np.ndarray, count: int) -> float:
        perturbed = wav.clone().reshape(n_clips, -1)
        for flat_idx in sequence[:count]:
            clip, frame = divmod(int(flat_idx), n_frames)
            lo = frame * samples_per_frame
            perturbed[clip, lo : lo + samples_per_frame] = 0.0
        return _logit(model, perturbed.reshape(wav.shape), clip_mask)

    def evaluate_keep(sequence: np.ndarray, count: int) -> float:
        perturbed = torch.zeros_like(wav).reshape(n_clips, -1)
        source = wav.clone().reshape(n_clips, -1)
        for flat_idx in sequence[:count]:
            clip, frame = divmod(int(flat_idx), n_frames)
            lo = frame * samples_per_frame
            perturbed[clip, lo : lo + samples_per_frame] = source[
                clip, lo : lo + samples_per_frame
            ]
        return _logit(model, perturbed.reshape(wav.shape), clip_mask)

    step = evaluate_keep if keep else evaluate
    fractions = np.linspace(0, 1, steps + 1)
    counts = (fractions * len(flat)).astype(int)

    return FaithfulnessCurve(
        fractions=fractions,
        scores=np.array([step(order, c) for c in counts]),
        random_scores=np.array([step(random_order, c) for c in counts]),
    )


def comprehensiveness(curve: FaithfulnessCurve) -> float:
    """Logit drop from removing the explanation's top regions.

    Higher is better: the explanation identified regions the model depended on.
    """
    return float(curve.scores[0] - curve.scores[-1])


def sufficiency(curve: FaithfulnessCurve) -> float:
    """Logit shortfall when only the top regions are kept.

    Lower is better: the explanation's regions alone nearly reproduce the call.
    """
    return float(curve.scores[0] - curve.scores[-1])


def is_faithful(curve: FaithfulnessCurve, margin: float = 0.0) -> bool:
    """Whether the explanation beats chance.

    This is the gate before any explanation figure goes into the report. An
    explanation that does not clear it should be reported as unfaithful rather
    than quietly dropped.
    """
    return curve.gap < -margin
