"""The objective must keep the primary head primary and ignore unlabelled rows."""

from __future__ import annotations

import pytest
import torch

from aimd.models.detector import DetectorOutput
from aimd.train.losses import DetectorLoss, class_weights


def make_output(batch: int = 4, n_sources: int | None = None) -> DetectorOutput:
    torch.manual_seed(0)
    return DetectorOutput(
        binary_logit=torch.randn(batch, requires_grad=True),
        embedding=torch.randn(batch, 8),
        layer_weights=torch.full((13,), 1 / 13),
        frame_weights=torch.full((batch, 5), 0.2),
        clip_weights=torch.full((batch, 2), 0.5),
        taxonomy_logits=torch.randn(batch, 4, requires_grad=True),
        source_logits=torch.randn(batch, n_sources, requires_grad=True) if n_sources else None,
    )


def make_batch(batch: int = 4) -> dict:
    return {
        "label": torch.tensor([0.0, 1.0, 1.0, 1.0][:batch]),
        "taxonomy": torch.tensor([0, 1, 2, 3][:batch]),
        "source": torch.tensor([-1, 0, 1, 0][:batch]),
    }


def test_class_weights_are_inverse_frequency_and_mean_one():
    w = class_weights(torch.tensor([800, 100, 60, 40]))
    assert w.argmax().item() == 3 and w.argmin().item() == 0
    torch.testing.assert_close(w.mean(), torch.tensor(1.0), rtol=1e-4, atol=1e-4)


def test_class_weights_clamp_extreme_imbalance():
    """An unclamped weight on a near-absent class destabilises training in a way
    that looks like a learning-rate problem."""
    assert class_weights(torch.tensor([100_000, 1]), clamp=10.0).max().item() <= 10.0


def test_class_weights_survive_an_absent_class():
    assert torch.isfinite(class_weights(torch.tensor([100, 0]))).all()


def test_auxiliary_head_does_not_dominate():
    """The reported metric comes from the binary head; the taxonomy term is a
    training signal, not the objective."""
    out, batch = make_output(), make_batch()
    binary_only = DetectorLoss(aux_weight=0.0)(out, batch)[1]["binary"]
    combined, parts = DetectorLoss(aux_weight=0.3)(out, batch)
    assert parts["binary"] == binary_only
    expected = binary_only + 0.3 * parts["taxonomy"]
    assert combined.item() == pytest.approx(expected, rel=1e-6)


def test_source_term_is_off_by_default():
    _, parts = DetectorLoss()(make_output(n_sources=3), make_batch())
    assert "source" not in parts


def test_source_term_ignores_rows_with_no_generator_label():
    """Real songs have no generator; they must not contribute to attribution."""
    out, batch = make_output(n_sources=3), make_batch()
    loss, parts = DetectorLoss(source_weight=1.0)(out, batch)
    assert "source" in parts and torch.isfinite(loss)

    all_unlabelled = {**batch, "source": torch.full_like(batch["source"], -1)}
    loss2, parts2 = DetectorLoss(source_weight=1.0)(out, all_unlabelled)
    assert torch.isfinite(loss2), "an all-unlabelled batch must not produce NaN"
    assert "source" not in parts2


def test_loss_is_differentiable():
    loss, _ = DetectorLoss()(make_output(), make_batch())
    loss.backward()


def test_pos_weight_shifts_the_binary_term():
    out, batch = make_output(), make_batch()
    plain = DetectorLoss()(out, batch)[1]["binary"]
    weighted = DetectorLoss(pos_weight=torch.tensor(3.0))(out, batch)[1]["binary"]
    assert plain != weighted
