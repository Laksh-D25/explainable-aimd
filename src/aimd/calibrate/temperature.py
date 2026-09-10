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
    """Lowest probability threshold whose false-positive rate is <= target.

    The operating point should be chosen against a false-positive budget rather
    than left at 0.5: the cost of wrongly flagging a human artist's track is not
    symmetric with the cost of missing an AI-generated one.
    """
    probs = scaler.transform(logits) if scaler is not None else 1 / (1 + np.exp(-logits))
    labels = np.asarray(labels).astype(int)
    real = probs[labels == 0]
    if len(real) == 0:
        return 0.5
    # The (1 - target) quantile of scores on real songs is the threshold that
    # leaves at most `target` of them above it.
    return float(np.quantile(real, 1.0 - target_fpr))
