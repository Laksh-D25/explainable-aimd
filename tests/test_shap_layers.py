"""Shapley attribution over MERT layers (XAI level 3).

Shapley values have exact mathematical properties -- efficiency, symmetry,
null-player -- so the implementation can be verified rather than eyeballed.
An attribution method that violates them is not an explanation.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from aimd.models.detector import Detector
from aimd.xai.shap_layers import (
    agreement_with_attention,
    coalition_values,
    layer_shapley,
    shapley_from_values,
)

N_LAYERS = 5  # 2^5 coalitions keeps the exact test fast


@pytest.fixture
def model_and_hidden(stub_backbone):
    torch.manual_seed(0)
    backbone = type(stub_backbone)(n_layers=N_LAYERS, dim=16)
    model = Detector(backbone=backbone).eval()
    hidden = torch.randn(1, 2, N_LAYERS, 7, 16)
    return model, hidden


def test_efficiency_shapley_values_sum_to_prediction_minus_baseline(model_and_hidden):
    """The defining property: contributions must account for the whole decision."""
    model, hidden = model_and_hidden
    values = coalition_values(model, hidden)
    phi = shapley_from_values(values, N_LAYERS)

    full = values[frozenset(range(N_LAYERS))]
    empty = values[frozenset()]
    assert phi.sum() == pytest.approx(full - empty, abs=1e-4)


def test_shapley_returns_one_value_per_layer(model_and_hidden):
    model, hidden = model_and_hidden
    assert layer_shapley(model, hidden).shape == (N_LAYERS,)


def test_a_duplicated_layer_gets_an_equal_share(model_and_hidden):
    """Symmetry: two layers carrying identical information must score equally."""
    model, hidden = model_and_hidden
    hidden[:, :, 1] = hidden[:, :, 0]  # layers 0 and 1 now identical
    phi = layer_shapley(model, hidden)
    assert phi[0] == pytest.approx(phi[1], abs=1e-4)


def test_permutation_sampling_approximates_the_exact_values(model_and_hidden):
    """The escape hatch for larger backbones must agree with exact Shapley."""
    model, hidden = model_and_hidden
    exact = layer_shapley(model, hidden)
    sampled = layer_shapley(model, hidden, n_permutations=200, seed=0)
    assert np.corrcoef(exact, sampled)[0, 1] > 0.9
    assert sampled.sum() == pytest.approx(exact.sum(), abs=1e-3)


def test_explaining_a_batch_is_refused(model_and_hidden):
    model, hidden = model_and_hidden
    with pytest.raises(ValueError, match="one song at a time"):
        coalition_values(model, hidden.repeat(2, 1, 1, 1, 1))


def test_agreement_with_attention_is_a_correlation():
    """Level 1 vs level 3: the faithfulness check, not a redundant metric."""
    # Strictly decreasing: ties would make a reversed ranking score above -1.
    aligned = np.array([0.5, 0.3, 0.1, 0.06, 0.04])
    assert agreement_with_attention(aligned, aligned) == pytest.approx(1.0)
    assert agreement_with_attention(aligned, aligned[::-1]) == pytest.approx(-1.0)

    # Must also accept tensors straight off DetectorOutput, grad and all.
    live = torch.tensor(aligned, requires_grad=True)
    assert agreement_with_attention(aligned, live) == pytest.approx(1.0)


def test_layer_mask_actually_restricts_aggregation(model_and_hidden):
    """The coalition mechanism itself: masked layers must not contribute."""
    model, hidden = model_and_hidden
    mask = torch.zeros(N_LAYERS, dtype=torch.bool)
    mask[:2] = True

    baseline = model.forward_from_hidden(hidden, layer_mask=mask).binary_logit
    perturbed = hidden.clone()
    perturbed[:, :, 2:] = torch.randn_like(perturbed[:, :, 2:])  # change only masked-out layers
    after = model.forward_from_hidden(perturbed, layer_mask=mask).binary_logit

    torch.testing.assert_close(baseline, after)


def test_masked_weights_still_form_a_distribution(model_and_hidden):
    model, hidden = model_and_hidden
    mask = torch.tensor([True, False, True, False, False])
    weights = model.forward_from_hidden(hidden, layer_mask=mask).layer_weights.detach()
    assert weights[~mask].abs().sum() == 0
    assert weights.sum().item() == pytest.approx(1.0, abs=1e-5)
