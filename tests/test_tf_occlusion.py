"""XAI level 2b: time-frequency occlusion.

The mechanism is verified against a model with a known frequency dependence,
because an attribution map that points at the wrong band is worse than none.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from aimd.models.detector import Detector
from aimd.xai.tf_occlusion import tf_occlusion_map

SR = 24_000


class BandDetector(nn.Module):
    """A stand-in whose logit depends only on energy in a known high band.

    This makes the attribution checkable: occluding that band must dominate the
    importance map, and occluding anything else must barely register.
    """

    def __init__(self, sample_rate: int = SR):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_layers, self.dim, self.max_seconds = 13, 32, 30.0

    def forward(self, wav, clip_mask=None, backbone_no_grad=True, **kw):
        # Windowed, to avoid spectral leakage smearing the band this stand-in keys on.
        window = torch.hann_window(1024, device=wav.device)
        spec = torch.stft(
            wav.flatten(0, 1), n_fft=1024, hop_length=256, window=window, return_complex=True
        )
        high = spec[:, 400:, :].abs().mean()  # ~9.4 kHz and above
        return type("Out", (), {"binary_logit": (high * 50).reshape(1)})()


@pytest.fixture
def tone_song():
    """One clip: broadband noise plus a strong 11 kHz tone."""
    t = torch.arange(SR) / SR
    torch.manual_seed(0)
    audio = 0.05 * torch.randn(SR) + 0.5 * torch.sin(2 * np.pi * 11_000 * t)
    return audio.reshape(1, 1, SR)


def test_map_shape_and_axes(tone_song):
    m = tf_occlusion_map(BandDetector(), tone_song, n_freq_bands=4, n_time_bands=5)
    assert m.importance.shape == (4, 5)
    assert len(m.freq_edges) == 5 and len(m.time_edges) == 6
    assert m.freq_edges[0] == 0
    assert m.freq_edges[-1] == pytest.approx(SR / 2, rel=1e-6)  # Nyquist
    assert m.time_edges[-1] == pytest.approx(1.0, rel=1e-6)


def test_attribution_finds_the_band_the_model_actually_uses(tone_song):
    """The decisive check: the map must point at the high band, not elsewhere."""
    m = tf_occlusion_map(BandDetector(), tone_song, n_freq_bands=4, n_time_bands=4)
    (f_lo, f_hi), _ = m.top_tile()
    assert f_lo >= SR / 2 * 0.5, f"expected a high-frequency tile, got {f_lo:.0f}-{f_hi:.0f} Hz"

    per_band = m.importance.mean(axis=1)
    assert per_band.argmax() >= 2, f"importance concentrated in the wrong band: {per_band}"


def test_occluding_an_irrelevant_band_barely_moves_the_logit(tone_song):
    m = tf_occlusion_map(BandDetector(), tone_song, n_freq_bands=4, n_time_bands=4)
    per_band = m.importance.mean(axis=1)
    assert abs(per_band[0]) < abs(per_band.max()) * 0.25


def test_baseline_survives_the_stft_roundtrip(tone_song):
    """Importance is measured against a resynthesised baseline, so STFT/ISTFT
    error is common to both terms rather than leaking into the attribution."""
    m = tf_occlusion_map(BandDetector(), tone_song, n_freq_bands=2, n_time_bands=2)
    direct = BandDetector().forward(tone_song).binary_logit.item()
    assert m.baseline_logit == pytest.approx(direct, rel=0.05)


def test_requires_a_single_song():
    with pytest.raises(ValueError, match="one song at a time"):
        tf_occlusion_map(BandDetector(), torch.randn(2, 1, SR))


def test_runs_on_the_real_detector_assembly(stub_backbone):
    """Shape compatibility with the actual model, not just the stand-in."""
    model = Detector(backbone=stub_backbone).eval()
    m = tf_occlusion_map(model, torch.randn(1, 2, SR), n_freq_bands=2, n_time_bands=3)
    assert m.importance.shape == (2, 3)
    assert np.isfinite(m.importance).all()
