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
