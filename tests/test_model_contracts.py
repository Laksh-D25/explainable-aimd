"""Contract tests for the detector's forward path.

These encode the architectural decisions that the review turned up, so a
regression on any of them is a test failure rather than a silent wrong number.
"""

from __future__ import annotations

import pytest
import torch

from aimd.models.detector import Detector
from aimd.models.heads import TAXONOMY_CLASSES
from aimd.models.pooling import AttentivePool, LayerAttention, LayerNormStack


def build(stub, **kw) -> Detector:
    return Detector(backbone=stub, **kw)


def test_forward_shapes(stub_backbone, noise):
    wav = noise(batch=2, clips=3, seconds=1.0)
    out = build(stub_backbone).forward(wav)

    assert out.binary_logit.shape == (2,)
    assert out.embedding.shape == (2, stub_backbone.dim)
    assert out.taxonomy_logits.shape == (2, len(TAXONOMY_CLASSES))
    assert out.clip_weights.shape == (2, 3)
    assert out.frame_weights.shape[0] == 2 * 3  # flattened B*C
    assert out.source_logits is None  # off unless n_sources is given


def test_bare_2d_input_is_treated_as_one_clip(stub_backbone, noise):
    wav = noise(batch=2, clips=1, seconds=1.0).squeeze(1)  # [B, N]
    out = build(stub_backbone).forward(wav)
    assert out.binary_logit.shape == (2,)
    assert out.clip_weights.shape == (2, 1)


def test_attention_weights_are_distributions(stub_backbone, noise):
    out = build(stub_backbone).forward(noise())
    for name, w in [
        ("layer", out.layer_weights),
        ("frame", out.frame_weights),
        ("clip", out.clip_weights),
    ]:
        assert torch.all(w >= 0), f"{name} weights must be non-negative"
        torch.testing.assert_close(
            w.sum(-1), torch.ones_like(w.sum(-1)), msg=f"{name} weights must sum to 1"
        )


def test_layer_weights_cover_all_13_states(stub_backbone, noise):
    """Aggregating across depth is the point -- a bug that silently uses one
    layer would still produce plausible losses, so pin the width here."""
    out = build(stub_backbone).forward(noise())
    assert out.layer_weights.shape[-1] == stub_backbone.n_layers == 13


def test_song_pool_masks_padded_clips(stub_backbone, noise):
    """Songs have different lengths, so clip counts vary within a batch.
    Padding clips must not leak into the song embedding."""
    wav = noise(batch=2, clips=4, seconds=1.0)
    mask = torch.tensor([[True, True, False, False], [True, True, True, True]])
    model = build(stub_backbone).eval()  # head dropout would mask the comparison

    out = model.forward(wav, clip_mask=mask)
    assert out.clip_weights[0, 2:].abs().sum() == 0

    # Changing only the padded clips must not change the first song's output.
    wav2 = wav.clone()
    wav2[0, 2:] = torch.randn_like(wav2[0, 2:])
    out2 = model.forward(wav2, clip_mask=mask)
    torch.testing.assert_close(out.binary_logit[0], out2.binary_logit[0])


def test_mean_pool_ablation_matches_plain_mean():
    pool = AttentivePool(dim=4, mode="mean")
    x = torch.randn(2, 5, 4)
    pooled, w = pool(x)
    torch.testing.assert_close(pooled, x.mean(dim=1))
    torch.testing.assert_close(w, torch.full((2, 5), 0.2))


def test_last_layer_ablation_selects_final_state(stub_backbone, noise):
    attn = LayerAttention(n_layers=13, dim=8, mode="last")
    hidden = torch.randn(2, 13, 7, 8)
    out, w = attn(hidden)
    torch.testing.assert_close(out, hidden[:, -1])
    assert w.argmax().item() == 12


def test_layernorm_stack_normalises_each_layer_separately():
    """Layers differ in scale by large factors; without per-layer normalisation
    the layer-attention softmax tracks magnitude instead of informativeness."""
    stack = LayerNormStack(n_layers=3, dim=16)
    hidden = torch.randn(2, 3, 5, 16)
    hidden[:, 1] *= 100.0  # blow up one layer's scale
    normed = stack(hidden)
    scales = normed.flatten(2).std(dim=-1).mean(dim=(0,))
    assert scales.max() / scales.min() < 1.5


@pytest.mark.parametrize("layer_mode", ["static", "conditioned", "last"])
@pytest.mark.parametrize("frame_pool", ["attn", "mean"])
@pytest.mark.parametrize("use_self_attn", [False, True])
def test_ablation_flags_all_run(stub_backbone, noise, layer_mode, frame_pool, use_self_attn):
    model = build(
        stub_backbone,
        layer_mode=layer_mode,
        frame_pool=frame_pool,
        use_self_attn=use_self_attn,
    )
    assert model.forward(noise()).binary_logit.shape == (2,)


def test_source_head_enabled_by_n_sources(stub_backbone, noise):
    out = build(stub_backbone, n_sources=5).forward(noise())
    assert out.source_logits.shape == (2, 5)


def test_padded_clips_do_not_reach_the_backbone(stub_backbone, noise):
    """Padded clips get zero weight at the song pool, so running the frozen
    backbone over them is wasted compute -- on a 4 GB GPU and a metered Kaggle
    quota that is worth avoiding."""
    calls = []
    original = stub_backbone.forward

    def counting_forward(wav, no_grad=False):
        calls.append(wav.shape[0])
        return original(wav, no_grad=no_grad)

    stub_backbone.forward = counting_forward
    try:
        model = build(stub_backbone).eval()
        wav = noise(batch=2, clips=4, seconds=1.0)
        mask = torch.tensor([[True, True, False, False], [True, False, False, False]])
        model.forward(wav, clip_mask=mask)
    finally:
        stub_backbone.forward = original

    assert calls == [3], f"expected only the 3 real clips through the backbone, got {calls}"


def test_skipping_padded_clips_does_not_change_the_result(stub_backbone, noise):
    """The optimisation must be exactly equivalent, not approximately."""
    model = build(stub_backbone).eval()
    wav = noise(batch=2, clips=4, seconds=1.0)
    mask = torch.tensor([[True, True, False, False], [True, True, True, True]])

    optimised = model.forward(wav, clip_mask=mask).binary_logit

    # Reference: run every clip through the backbone, then pool with the mask.
    hidden = stub_backbone(wav.flatten(0, 1), no_grad=True)
    reference = model.forward_from_hidden(
        hidden.view(*wav.shape[:2], *hidden.shape[1:]), mask
    ).binary_logit

    torch.testing.assert_close(optimised, reference)
