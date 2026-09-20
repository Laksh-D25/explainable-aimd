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


def rms_normalize(wav: np.ndarray, target_rms: float = 0.05,
                  max_gain: float = 20.0) -> np.ndarray:
    """Scale to a fixed RMS (loudness) rather than a fixed peak.

    Peak normalisation leaves a large shortcut in this task: commercial music is
    mastered loud and heavily compressed, while generated tracks are not, so RMS
    alone separates the classes at 0.81 AUROC even after peak normalisation. A
    detector will take that cue, and its score then reflects mastering practice
    rather than synthesis artefacts.

    Gain is capped so a near-silent clip is not amplified into noise.
    """
    rms = float(np.sqrt(np.mean(wav.astype(np.float64) ** 2)))
    if rms < 1e-8:
        return wav.astype(np.float32)
    gain = min(target_rms / rms, max_gain)
    out = wav * gain
    # Re-limit rather than clip hard: a scaled-up loud track can exceed 1.0.
    peak = np.abs(out).max()
    if peak > 0.99:
        out = out * (0.99 / peak)
    return out.astype(np.float32)


def randomize_dynamics(wav: np.ndarray, rng: np.random.Generator | None = None,
                       low: float = 0.55, high: float = 1.45) -> np.ndarray:
    """Randomise dynamic range, then restore loudness.

    Crest factor (peak / RMS) separates the classes at 0.81 AUROC and survives
    every amplitude normalisation, because it is scale-invariant: commercial
    releases are limited and compressed, generated tracks are not. A detector
    will read mastering practice instead of synthesis artefacts.

    Applying `sign(x) * |x| ** gamma` with a random gamma compresses (gamma < 1)
    or expands (gamma > 1) the range, so the cue is smeared across both classes
    rather than tracking the label. RMS is restored afterwards so the transform
    does not reintroduce a loudness difference.

    Applied to *both* classes and to eval as well as train: the goal is to
    remove a confound from the task, not to augment one split.
    """
    rng = rng or np.random.default_rng()
    gamma = float(rng.uniform(low, high))
    shaped = np.sign(wav) * np.abs(wav) ** gamma
    return rms_normalize(shaped)


def fit_length(wav: np.ndarray, n_samples: int) -> np.ndarray:
    """Trim or zero-pad to exactly `n_samples`.

    Perturbations do not all preserve length -- MP3 encoding in particular adds
    encoder padding, so a round-trip returns slightly more samples than it was
    given. The model needs fixed-length clips, so length is pinned after any
    perturbation rather than assumed.
    """
    if len(wav) == n_samples:
        return wav.astype(np.float32)
    if len(wav) > n_samples:
        return wav[:n_samples].astype(np.float32)
    out = np.zeros(n_samples, dtype=np.float32)
    out[: len(wav)] = wav
    return out


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


def spectral_rolloff(wav: np.ndarray, sample_rate: int, percentile: float = 0.99) -> float:
    """Frequency (Hz) below which `percentile` of spectral energy lies.

    Used to detect a bandwidth mismatch between classes -- see
    `bandwidth_report`.
    """
    spectrum = np.abs(np.fft.rfft(wav * np.hanning(len(wav)))) ** 2
    freqs = np.fft.rfftfreq(len(wav), 1 / sample_rate)
    total = spectrum.sum()
    if total <= 0:
        return 0.0
    return float(freqs[np.searchsorted(np.cumsum(spectrum), percentile * total)])


def bandwidth_report(
    manifest,
    n_per_class: int = 40,
    sample_rate: int = TARGET_SAMPLE_RATE,
    seconds: float = 5.0,
    seed: int = 0,
) -> dict:
    """Compare spectral bandwidth between the real and fake classes.

    A detector trained on classes that differ in bandwidth learns the codec
    chain, not the generator. The failure is invisible in the metrics -- it
    produces an *excellent* score -- so it has to be checked directly.

    The case that motivated this: FakeMusicCaps generated audio is 16 kHz,
    while MusicCaps real audio is 48 kHz. Resampled to MERT's 24 kHz, the
    generated clips carry nothing above 8 kHz and the real ones reach 12 kHz.
    Separating those is trivial and meaningless. The fix is to band-limit the
    real audio to the generated audio's rate *before* resampling, so both
    classes traverse the same chain.

    Returns median rolloff per class and the ratio between them. A ratio far
    from 1.0 means the classes are distinguishable on bandwidth alone.
    """
    rng = np.random.default_rng(seed)
    out: dict = {}
    for label, name in ((0, "real"), (1, "fake")):
        rows = manifest[manifest["label"] == label]
        if rows.empty:
            continue
        take = rows.iloc[rng.permutation(len(rows))[:n_per_class]]
        values = []
        for path in take["path"]:
            try:
                wav = load_audio(path, sample_rate)[: int(seconds * sample_rate)]
                if len(wav) > 1024:
                    values.append(spectral_rolloff(wav, sample_rate))
            except Exception:
                continue
        if values:
            out[f"{name}_rolloff_hz"] = float(np.median(values))
            out[f"{name}_n"] = len(values)

    if "real_rolloff_hz" in out and "fake_rolloff_hz" in out:
        lo, hi = sorted((out["real_rolloff_hz"], out["fake_rolloff_hz"]))
        out["ratio"] = hi / lo if lo > 0 else float("inf")
        out["suspicious"] = out["ratio"] > 1.15
    return out
