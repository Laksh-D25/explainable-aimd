"""Temperature scaling.

The architecture diagram labelled its output "Calibrated" but contained no
calibration stage. This is that stage: a single scalar T dividing the logit,
fitted on validation by minimising NLL.

It is deliberately the simplest method that works. Temperature scaling cannot
change the ranking of predictions, so F1, AUROC and EER are untouched -- it
only makes the probability mean what it says. That property is why it is safe
to apply after the fact, and why the decision threshold should be chosen on the
calibrated scale.
"""

from __future__ import annotations

import numpy as np
import torch


class TemperatureScaler:
    """Fit on validation logits, apply to test logits."""

    def __init__(self, temperature: float = 1.0):
        self.temperature = float(temperature)

    def fit(self, logits: np.ndarray, labels: np.ndarray, max_iter: int = 200) -> "TemperatureScaler":
        logit_t = torch.as_tensor(np.asarray(logits, dtype=np.float64))
        label_t = torch.as_tensor(np.asarray(labels, dtype=np.float64))
        # Optimise log T so temperature stays strictly positive.
        log_t = torch.zeros(1, dtype=torch.float64, requires_grad=True)
        optimizer = torch.optim.LBFGS([log_t], lr=0.1, max_iter=max_iter)

        def closure():
            optimizer.zero_grad()
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logit_t / log_t.exp(), label_t
            )
            loss.backward()
            # LBFGS calls float() on the return value; handing back a tensor
            # that still carries grad triggers a warning on every calibration.
            return loss.detach()

        optimizer.step(closure)
        self.temperature = float(log_t.exp().item())
        return self

    def transform(self, logits: np.ndarray) -> np.ndarray:
        """Calibrated probabilities."""
        return 1 / (1 + np.exp(-np.asarray(logits, dtype=np.float64) / self.temperature))

    def fit_transform(self, logits: np.ndarray, labels: np.ndarray) -> np.ndarray:
        return self.fit(logits, labels).transform(logits)


def threshold_for_target_fpr(
    logits: np.ndarray, labels: np.ndarray, target_fpr: float = 0.01, scaler=None
) -> float:
    """Operating point chosen against a false-positive budget.

    The cost of wrongly flagging a human artist's track is not symmetric with
    the cost of missing an AI-generated one, so the threshold comes from an FPR
    budget rather than sitting at 0.5.

    **Perfect separation needs different handling.** When validation separates
    completely -- which it does here, at F1 1.000 -- every real score sits near
    zero, so the (1 - target) quantile of them is also near zero. That threshold
    is technically valid on validation and catastrophic anywhere else: applied
    to unseen generators it labelled *everything* fake (FPR 1.00), producing an
    F1 identical to a trivial all-positive baseline.

    With a clean margin, any threshold inside it scores identically on
    validation, so the quantile is an arbitrary pick from a wide interval. The
    midpoint of the margin is the robust choice: it is the point furthest from
    both classes, and therefore the one most likely to survive a distribution
    shift.
    """
    probs = scaler.transform(logits) if scaler is not None else 1 / (1 + np.exp(-logits))
    labels = np.asarray(labels).astype(int)
    real, fake = probs[labels == 0], probs[labels == 1]
    if len(real) == 0:
        return 0.5

    if len(fake) and real.max() < fake.min():
        # Separable: sit in the middle of the gap rather than hard against the
        # real class, where any shift pushes every sample across the boundary.
        return float((real.max() + fake.min()) / 2)

    return float(np.quantile(real, 1.0 - target_fpr))
