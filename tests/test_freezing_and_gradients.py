"""Frozen backbone must still pass gradients.

Wrapping the backbone in a permanent `no_grad` is the obvious way to freeze it
and it silently breaks Grad-CAM, the occlusion attributions and the
faithfulness metrics -- all of which need a backward pass reaching the audio.
Freezing is `requires_grad_(False)` on parameters; these tests hold that line.
"""

from __future__ import annotations

import torch

from aimd.models.detector import Detector


def test_backbone_parameters_are_frozen(stub_backbone):
    model = Detector(backbone=stub_backbone)
    assert all(not p.requires_grad for p in model.backbone.parameters())
    # ...but the trainable tail is trainable.
    assert any(p.requires_grad for p in model.trainable_parameters())
    assert all(p.requires_grad for p in model.binary_head.parameters())


def test_optimiser_sees_only_the_tail(stub_backbone):
    model = Detector(backbone=stub_backbone)
    n_trainable = sum(p.numel() for p in model.trainable_parameters())
    n_backbone = sum(p.numel() for p in model.backbone.parameters())
    assert n_trainable > 0 and n_backbone > 0
    assert not any(
        p is q for p in model.trainable_parameters() for q in model.backbone.parameters()
    )


def test_gradients_reach_the_audio_when_requested(stub_backbone, noise):
    """The XAI path: backbone_no_grad=False must give d(logit)/d(waveform)."""
    model = Detector(backbone=stub_backbone)
    wav = noise(batch=1, clips=1, seconds=1.0).requires_grad_(True)

    out = model.forward(wav, backbone_no_grad=False)
    out.binary_logit.sum().backward()

    assert wav.grad is not None
    assert wav.grad.abs().sum() > 0, "no gradient reached the input audio"


def test_training_fast_path_detaches_the_backbone(stub_backbone, noise):
    model = Detector(backbone=stub_backbone)
    wav = noise(batch=1, clips=1, seconds=1.0).requires_grad_(True)

    out = model.forward(wav, backbone_no_grad=True)
    out.binary_logit.sum().backward()

    assert wav.grad is None or wav.grad.abs().sum() == 0
    assert model.binary_head.net[1].weight.grad is not None  # tail still learns


def test_mert_backbone_train_keeps_inner_model_in_eval():
    """Detector.train() must not flip MERT into training mode -- dropout inside
    a frozen backbone would make its features non-deterministic across epochs.

    This exercises the real MertBackbone.train override as an unbound method,
    so it holds the production code to the contract without downloading weights.
    """
    import torch.nn as nn

    from aimd.models.backbone import MertBackbone

    class Shim(MertBackbone):
        """Subclasses MertBackbone but skips __init__ (which downloads weights),
        so the inherited train() is the real one under test."""

        def __init__(self):
            nn.Module.__init__(self)
            self.model = nn.Linear(2, 2)

    shim = Shim()
    shim.train(True)
    assert shim.training is True, "the wrapper should follow the requested mode"
    assert shim.model.training is False, "the HF model must stay in eval"


def test_detector_train_mode_reaches_the_tail(stub_backbone):
    model = Detector(backbone=stub_backbone)
    model.train()
    assert model.binary_head.training
    model.eval()
    assert not model.binary_head.training
