"""Segmentation contracts.

Clip selection is a correctness concern, not a detail: eval scores must be a
property of the song rather than of which clips were drawn, and padding must
never reach the song embedding.
"""

from __future__ import annotations

import numpy as np
import pytest

from aimd.data.audio import clip_offsets, pad_clips, peak_normalize, segment

SR = 24_000


def song(seconds: float) -> np.ndarray:
    return np.random.default_rng(0).standard_normal(int(seconds * SR)).astype(np.float32)


def test_uniform_segmentation_is_deterministic():
    """Two evaluations of one song must give identical clips, or the reported
    metric changes between runs of the same checkpoint."""
    wav = song(60)
    a = segment(wav, clip_seconds=10, n_clips=4, strategy="uniform")
    b = segment(wav, clip_seconds=10, n_clips=4, strategy="uniform")
    np.testing.assert_array_equal(a, b)


def test_uniform_segmentation_spans_the_whole_song():
    """Artefacts may sit anywhere, so eval coverage must reach the final second."""
    offsets = clip_offsets(60 * SR, 10 * SR, n_clips=4, strategy="uniform", rng=None)
    assert offsets[0] == 0
    assert offsets[-1] == 60 * SR - 10 * SR


def test_random_segmentation_varies_but_is_seedable():
    wav = song(60)
    a = segment(wav, 10, 4, strategy="random", rng=np.random.default_rng(1))
    b = segment(wav, 10, 4, strategy="random", rng=np.random.default_rng(1))
    c = segment(wav, 10, 4, strategy="random", rng=np.random.default_rng(2))
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)


def test_clip_shape_is_exact():
    clips = segment(song(37.5), clip_seconds=10, n_clips=8)
    assert clips.shape[1] == 10 * SR


def test_song_shorter_than_one_clip_is_padded_not_dropped():
    """SONICS starts at 32 s but FakeMusicCaps clips are ~10 s; a short song
    must survive segmentation rather than produce an empty batch."""
    clips = segment(song(4), clip_seconds=10, n_clips=8)
    assert clips.shape == (1, 10 * SR)
    assert clips[0, 4 * SR :].sum() == 0  # tail is zero padding


def test_pad_clips_masks_the_padding():
    clips = segment(song(25), clip_seconds=10, n_clips=3)
    padded, mask = pad_clips(clips, max_clips=8)
    assert padded.shape == (8, 10 * SR)
    assert mask.sum() == len(clips)
    assert padded[mask.sum() :].sum() == 0


def test_pad_clips_truncates_when_over_budget():
    clips = segment(song(240), clip_seconds=10, n_clips=20)
    padded, mask = pad_clips(clips, max_clips=12)
    assert padded.shape[0] == 12 and mask.all()


def test_peak_normalize_scales_to_target_and_survives_silence():
    np.testing.assert_allclose(np.abs(peak_normalize(song(1) * 0.01)).max(), 0.95, rtol=1e-5)
    silent = np.zeros(1000, dtype=np.float32)
    np.testing.assert_array_equal(peak_normalize(silent), silent)


def test_unknown_strategy_is_rejected():
    with pytest.raises(ValueError, match="unknown segmentation strategy"):
        clip_offsets(SR * 60, SR * 10, 4, "sliding", None)


def test_fit_length_trims_and_pads():
    from aimd.data.audio import fit_length

    assert len(fit_length(np.zeros(100, dtype=np.float32), 50)) == 50
    padded = fit_length(np.ones(30, dtype=np.float32), 50)
    assert len(padded) == 50 and padded[30:].sum() == 0
    assert len(fit_length(np.zeros(50, dtype=np.float32), 50)) == 50


# --- bandwidth confound -------------------------------------------------------


def _write(path, wav, sr=SR):
    import soundfile as sf
    sf.write(path, wav.astype(np.float32), sr)


def _band_limited(seconds, cutoff_hz, sr=SR, seed=0):
    """Noise with energy only below `cutoff_hz` -- stands in for a codec chain."""
    rng = np.random.default_rng(seed)
    n = int(seconds * sr)
    spec = np.fft.rfft(rng.standard_normal(n))
    spec[np.fft.rfftfreq(n, 1 / sr) > cutoff_hz] = 0
    wav = np.fft.irfft(spec, n)
    return wav / (np.abs(wav).max() + 1e-9) * 0.5


def test_spectral_rolloff_tracks_the_cutoff():
    from aimd.data.audio import spectral_rolloff

    low = spectral_rolloff(_band_limited(2.0, 4000), SR)
    high = spectral_rolloff(_band_limited(2.0, 10000), SR)
    assert low < 5000 < high


def test_bandwidth_report_flags_a_mismatch(tmp_path):
    """The case that motivated this: 16 kHz generated audio against 48 kHz real
    audio. A detector separating those reads the codec chain, not the generator,
    and the metrics look excellent while measuring nothing."""
    import pandas as pd

    from aimd.data.audio import bandwidth_report

    rows = []
    for i in range(6):
        for label, cutoff in ((0, 11000), (1, 7000)):  # real wideband, fake band-limited
            p = tmp_path / f"{label}_{i}.wav"
            _write(p, _band_limited(2.0, cutoff, seed=i * 2 + label))
            rows.append({"path": str(p), "label": label})

    report = bandwidth_report(pd.DataFrame(rows), n_per_class=6, seconds=1.5)
    assert report["suspicious"] is True
    assert report["ratio"] > 1.15
    assert report["real_rolloff_hz"] > report["fake_rolloff_hz"]


def test_bandwidth_report_passes_matched_classes(tmp_path):
    import pandas as pd

    from aimd.data.audio import bandwidth_report

    rows = []
    for i in range(6):
        for label in (0, 1):
            p = tmp_path / f"{label}_{i}.wav"
            _write(p, _band_limited(2.0, 9000, seed=i * 2 + label))
            rows.append({"path": str(p), "label": label})

    report = bandwidth_report(pd.DataFrame(rows), n_per_class=6, seconds=1.5)
    assert report["suspicious"] is False


def test_bandwidth_report_handles_a_missing_class(tmp_path):
    """A manifest with no real class (FakeMusicCaps alone) must not crash."""
    import pandas as pd

    from aimd.data.audio import bandwidth_report

    p = tmp_path / "only_fake.wav"
    _write(p, _band_limited(2.0, 8000))
    report = bandwidth_report(pd.DataFrame([{"path": str(p), "label": 1}]), n_per_class=2)
    assert "ratio" not in report
