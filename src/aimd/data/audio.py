"""Audio loading and segmentation.

This is the diagram's "Pre-processing" box (resample / segment / normalise),
with one addition it left implicit: segmentation has to be *strategy-aware*.

SONICS songs run 32-240 s and MERT is a 5 s model, so a song is always seen as
a set of clips. How those clips are chosen differs by split:

* **train** -- random offsets, so the model sees different parts each epoch and
  the clip choice acts as augmentation in its own right.
* **eval** -- uniformly spaced and deterministic, so a score is a property of
  the song rather than of which clips happened to be drawn. Without this,
  repeated evaluation of the same checkpoint gives different numbers.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

TARGET_SAMPLE_RATE = 24_000  # MERT-v1-95M


def load_audio(path: str | Path, target_sr: int = TARGET_SAMPLE_RATE) -> np.ndarray:
    """Load any supported file as mono float32 at `target_sr`."""
    import torchaudio

    wav, sr = torchaudio.load(str(path))  # [channels, samples]
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    return wav.squeeze(0).numpy().astype(np.float32)


def peak_normalize(wav: np.ndarray, peak: float = 0.95) -> np.ndarray:
    """Scale to a fixed peak.

    Applied before augmentation so that gain and codec conditions start from a
    consistent level; MERT does its own zero-mean/unit-variance normalisation
    afterwards, so this is about making perturbations comparable across songs,
    not about matching the backbone's expectations.
    """
    m = np.abs(wav).max()
    return wav if m < 1e-8 else (wav * (peak / m)).astype(np.float32)


def clip_offsets(
    n_samples: int, clip_samples: int, n_clips: int, strategy: str, rng: np.random.Generator | None
) -> list[int]:
    """Start offsets for `n_clips` clips, per strategy.

    Songs shorter than one clip yield a single offset at 0 and are zero-padded
    by `segment`; the caller masks them via the returned clip mask.
    """
    if strategy not in {"uniform", "random"}:
        raise ValueError(f"unknown segmentation strategy: {strategy!r}")
    span = n_samples - clip_samples
    if span <= 0:
        return [0]
    if strategy == "uniform":
        # Evenly spaced and deterministic: covers the whole song and makes a
        # song's score reproducible across runs.
        return [int(round(x)) for x in np.linspace(0, span, num=n_clips)]
    if rng is None:
        rng = np.random.default_rng()
    return sorted(int(x) for x in rng.integers(0, span + 1, size=n_clips))


def segment(
    wav: np.ndarray,
    clip_seconds: float = 10.0,
    n_clips: int = 8,
    sample_rate: int = TARGET_SAMPLE_RATE,
    strategy: str = "uniform",
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Cut a waveform into clips -> [n_actual_clips, clip_samples], zero-padded."""
    clip_samples = int(clip_seconds * sample_rate)
    offsets = clip_offsets(len(wav), clip_samples, n_clips, strategy, rng)
    clips = np.zeros((len(offsets), clip_samples), dtype=np.float32)
    for i, start in enumerate(offsets):
        chunk = wav[start : start + clip_samples]
        clips[i, : len(chunk)] = chunk
    return clips


def pad_clips(clips: np.ndarray, max_clips: int) -> tuple[np.ndarray, np.ndarray]:
    """Pad a clip stack to `max_clips` -> (clips, mask).

    Songs differ in length so clip counts differ within a batch. The mask is
    what stops padding from reaching the song embedding -- `AttentivePool`
    assigns padded positions exactly zero weight.
    """
    n, clip_samples = clips.shape
    if n > max_clips:
        return clips[:max_clips], np.ones(max_clips, dtype=bool)
    padded = np.zeros((max_clips, clip_samples), dtype=np.float32)
    padded[:n] = clips
    mask = np.zeros(max_clips, dtype=bool)
    mask[:n] = True
    return padded, mask
