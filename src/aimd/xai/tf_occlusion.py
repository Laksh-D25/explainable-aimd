"""XAI level 2b: true time-frequency attribution by occlusion.

Level 2a answers *when*. This answers *at which frequency*, which the
architecture asked for but Grad-CAM cannot deliver on a waveform model.

The method sidesteps the missing spectrogram entirely instead of pretending it
exists: take the STFT, zero one time-frequency tile, resynthesise audio, and
measure how far the logit moves. Because the perturbation happens in the STFT
domain and the model still sees a waveform, the attribution is genuinely about
frequency content rather than about conv channels.

The cost is real -- one backbone forward per tile -- so this is an audit-subset
method by design, not something to run over a test set. A 6x8 grid is 48
forward passes per song; level 2a is the one that runs everywhere.

Occlusion also has a property gradients lack: it measures what the model does
when evidence is actually absent, rather than the local slope around the input.
For narrowband artefacts of the kind Afchar et al. describe, that is the more
meaningful question.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ..models.detector import Detector


@dataclass
class TimeFrequencyMap:
    """Logit drop caused by occluding each tile. Higher = more important."""

    importance: np.ndarray  # [n_freq_bands, n_time_bands]
    freq_edges: np.ndarray  # Hz, length n_freq_bands + 1
    time_edges: np.ndarray  # seconds, length n_time_bands + 1
    baseline_logit: float

    def top_tile(self) -> tuple[tuple[float, float], tuple[float, float]]:
        """((f_lo, f_hi), (t_lo, t_hi)) of the most important tile."""
        f, t = np.unravel_index(np.argmax(self.importance), self.importance.shape)
        return (
            (float(self.freq_edges[f]), float(self.freq_edges[f + 1])),
            (float(self.time_edges[t]), float(self.time_edges[t + 1])),
        )


@torch.no_grad()
def tf_occlusion_map(
    model: Detector,
    wav: torch.Tensor,
    clip_mask: torch.Tensor | None = None,
    n_freq_bands: int = 6,
    n_time_bands: int = 8,
    n_fft: int = 1024,
    hop_length: int = 256,
    sample_rate: int = 24_000,
) -> TimeFrequencyMap:
    """Occlude each time-frequency tile of one song and record the logit drop.

    Args:
        wav: [1, C, N] one song's clips.

    Returns:
        A TimeFrequencyMap whose importance is `baseline - occluded` logit, so
        positive values mark tiles the fake-class decision depended on.
    """
    if wav.dim() != 3 or wav.shape[0] != 1:
        raise ValueError(f"explain one song at a time: got {tuple(wav.shape)}")

    device = wav.device
    clips = wav[0]  # [C, N]
    n_clips, n_samples = clips.shape
    window = torch.hann_window(n_fft, device=device)

    spec = torch.stft(
        clips, n_fft=n_fft, hop_length=hop_length, window=window, return_complex=True
    )  # [C, F, T]
    n_freq, n_time = spec.shape[1], spec.shape[2]

    def resynthesise(modified: torch.Tensor) -> torch.Tensor:
        audio = torch.istft(
            modified, n_fft=n_fft, hop_length=hop_length, window=window, length=n_samples
        )
        return audio.unsqueeze(0)

    baseline = float(model.forward(resynthesise(spec), clip_mask=clip_mask).binary_logit.item())

    freq_bounds = np.linspace(0, n_freq, n_freq_bands + 1).astype(int)
    time_bounds = np.linspace(0, n_time, n_time_bands + 1).astype(int)
    importance = np.zeros((n_freq_bands, n_time_bands))

    for i in range(n_freq_bands):
        for j in range(n_time_bands):
            occluded = spec.clone()
            occluded[:, freq_bounds[i] : freq_bounds[i + 1], time_bounds[j] : time_bounds[j + 1]] = 0
            logit = float(
                model.forward(resynthesise(occluded), clip_mask=clip_mask).binary_logit.item()
            )
            importance[i, j] = baseline - logit

    return TimeFrequencyMap(
        importance=importance,
        freq_edges=freq_bounds / n_freq * (sample_rate / 2),
        time_edges=time_bounds / n_time * (n_samples / sample_rate),
        baseline_logit=baseline,
    )
