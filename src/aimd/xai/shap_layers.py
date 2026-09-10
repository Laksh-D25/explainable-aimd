"""XAI level 3: Shapley attribution over MERT's 13 layers.

The coalition game is natural here. A "player" is one of MERT's hidden states,
and the value of a coalition S is the detector's logit when layer attention is
restricted to S and renormalised. Shapley values then say how much each layer
*actually moved this decision*.

That is a different question from level 1. The learned attention weights say
which layers the model allocates weight to in general; these say which layers
changed the outcome for this song. Where the two disagree, the attention
weights are not a faithful explanation -- which is a finding worth reporting,
given that the explainability literature this project builds on notes that
explanation fidelity remains an open metric.

Cost is modest because the frozen backbone runs once: coalitions re-evaluate
only the ~0.6 M-parameter tail. With 13 layers, exact enumeration is 2^13 =
8192 tail passes, so exact Shapley values are affordable rather than estimated.
"""

from __future__ import annotations

from itertools import combinations
from math import comb

import numpy as np
import torch

from ..models.detector import Detector


@torch.no_grad()
def coalition_values(
    model: Detector,
    hidden: torch.Tensor,
    clip_mask: torch.Tensor | None = None,
    subsets: list[tuple[int, ...]] | None = None,
) -> dict[frozenset[int], float]:
    """Evaluate v(S) -- the binary logit under each layer coalition.

    `hidden` is a single song: [1, C, L, T, D].
    """
    if hidden.shape[0] != 1:
        raise ValueError("explain one song at a time; got batch of " f"{hidden.shape[0]}")
    n_layers = hidden.shape[2]
    if subsets is None:
        subsets = [
            s for r in range(n_layers + 1) for s in combinations(range(n_layers), r)
        ]

    values: dict[frozenset[int], float] = {}
    for subset in subsets:
        key = frozenset(subset)
        if key in values:
            continue
        if not subset:
            # The empty coalition contributes no features. Defined as the logit
            # of a zeroed representation so that the Shapley values sum to
            # (full prediction - baseline), the standard efficiency property.
            zeros = torch.zeros_like(hidden)
            values[key] = float(model.forward_from_hidden(zeros, clip_mask).binary_logit.item())
            continue
        mask = torch.zeros(n_layers, dtype=torch.bool, device=hidden.device)
        mask[list(subset)] = True
        out = model.forward_from_hidden(hidden, clip_mask, layer_mask=mask)
        values[key] = float(out.binary_logit.item())
    return values


def shapley_from_values(values: dict[frozenset[int], float], n_layers: int) -> np.ndarray:
    """Exact Shapley values from a complete coalition table."""
    phi = np.zeros(n_layers)
    for i in range(n_layers):
        others = [j for j in range(n_layers) if j != i]
        for r in range(len(others) + 1):
            weight = 1.0 / (n_layers * comb(n_layers - 1, r))
            for subset in combinations(others, r):
                key = frozenset(subset)
                phi[i] += weight * (values[key | {i}] - values[key])
    return phi


def layer_shapley(
    model: Detector,
    hidden: torch.Tensor,
    clip_mask: torch.Tensor | None = None,
    n_permutations: int | None = None,
    seed: int = 0,
) -> np.ndarray:
    """Shapley value per MERT layer for one song.

    Exact by default (2^L coalitions). Pass `n_permutations` to use Monte-Carlo
    permutation sampling instead, which is the escape hatch if the layer count
    grows -- MERT-330M has 25 hidden states, where exact enumeration is 33 M
    coalitions and no longer affordable.
    """
    n_layers = hidden.shape[2]
    if n_permutations is None:
        return shapley_from_values(coalition_values(model, hidden, clip_mask), n_layers)

    rng = np.random.default_rng(seed)
    phi = np.zeros(n_layers)
    for _ in range(n_permutations):
        order = rng.permutation(n_layers)
        prefix: list[int] = []
        needed = [tuple(sorted(prefix))]
        for layer in order:
            prefix.append(int(layer))
            needed.append(tuple(sorted(prefix)))
        values = coalition_values(model, hidden, clip_mask, subsets=needed)
        prev = values[frozenset()]
        prefix = []
        for layer in order:
            prefix.append(int(layer))
            current = values[frozenset(prefix)]
            phi[layer] += current - prev
            prev = current
    return phi / n_permutations


def _to_numpy(x) -> np.ndarray:
    """Accept live model outputs, which are tensors that still carry grad."""
    if torch.is_tensor(x):
        return x.detach().cpu().numpy().ravel()
    return np.asarray(x).ravel()


def agreement_with_attention(shapley, attention) -> float:
    """Spearman correlation between level 3 and level 1 layer rankings.

    Low agreement means the model's own attention weights do not reflect what
    drove the decision -- report it, do not hide it.
    """
    from scipy.stats import spearmanr

    return float(spearmanr(_to_numpy(shapley), _to_numpy(attention)).statistic)
