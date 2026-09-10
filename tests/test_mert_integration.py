"""Integration tests against the real MERT checkpoint.

Opt-in (`pytest -m real_mert`) because they download ~400 MB. They pin the
facts the architecture review depends on -- if MERT's shape or frame rate ever
differs from what the design assumes, the failure should surface here rather
than as a confusing training result.
"""

from __future__ import annotations

import pytest
import torch

from aimd.models.backbone import MERT_FRAME_RATE, MERT_SAMPLE_RATE, MertBackbone
from aimd.models.detector import Detector

pytestmark = pytest.mark.real_mert


@pytest.fixture(scope="module")
def mert() -> MertBackbone:
    return MertBackbone()


def test_backbone_geometry_matches_the_design(mert):
    assert mert.n_layers == 13, "the layer-attention stage is sized for 13 hidden states"
    assert mert.dim == 768
    assert mert.sample_rate == MERT_SAMPLE_RATE


def test_frame_rate_is_75hz(mert):
    """The 75 Hz rate is what makes full-song input infeasible: a 120 s song is
    ~9000 frames and MERT's attention is O(T^2). This is the measurement behind
    the decision to segment into clips and aggregate at song level."""
    seconds = 5
    hidden = mert(torch.randn(1, seconds * mert.sample_rate), no_grad=True)
    frames_per_second = hidden.shape[2] / seconds
    assert abs(frames_per_second - MERT_FRAME_RATE) < 2, f"got {frames_per_second:.1f} Hz"


def test_hidden_state_shape(mert):
    hidden = mert(torch.randn(2, 5 * mert.sample_rate), no_grad=True)
    assert hidden.shape[:2] == (2, 13)
    assert hidden.shape[3] == 768


def test_long_input_warns_rather_than_silently_exploding(mert):
    """A whole song must not be fed to the backbone by accident."""
    with pytest.warns(UserWarning, match="segment into clips"):
        mert(torch.randn(1, int(mert.max_seconds + 5) * mert.sample_rate), no_grad=True)


def test_only_the_tail_is_trainable(mert):
    model = Detector(backbone=mert)
    trainable = sum(p.numel() for p in model.trainable_parameters())
    frozen = sum(p.numel() for p in model.backbone.parameters())
    assert frozen > 90e6, "MERT-95M should contribute ~94 M frozen parameters"
    assert trainable < 5e6, "the head should stay small enough to train on the dev subset"


def test_gradients_reach_audio_through_real_mert(mert):
    """Grad-CAM, occlusion and the faithfulness metrics all depend on this."""
    model = Detector(backbone=mert).eval()
    wav = torch.randn(1, 1, 5 * mert.sample_rate, requires_grad=True)
    model.forward(wav, backbone_no_grad=False).binary_logit.sum().backward()
    assert wav.grad is not None and wav.grad.abs().sum() > 0


def test_conv_module_output_is_time_by_channel_not_time_by_frequency(mert):
    """Pins the reason Level 2 was split in two.

    MERT consumes raw waveform through a 1-D conv stack, so the Grad-CAM target
    layer emits (channel, time) -- there is no frequency axis anywhere in the
    forward pass. Attributions taken here localise *when*, not *at which
    frequency*; true time-frequency evidence needs the STFT-occlusion path.
    """
    wav = torch.randn(1, 5 * mert.sample_rate)
    with torch.no_grad():
        conv_out = mert.conv_module(mert.normalize(wav))
    assert conv_out.dim() == 3
    _, channels, frames = conv_out.shape
    assert channels == 512, "Hubert-style conv stack emits 512 channels, not frequency bins"
    assert abs(frames / 5 - MERT_FRAME_RATE) < 2
