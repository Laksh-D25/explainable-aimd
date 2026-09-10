"""XAI level 2a: temporal relevance over the waveform.

The architecture called for "Grad-CAM Spectrogram Localization". That is not
achievable as drawn: MERT consumes raw 24 kHz audio through a 1-D convolutional
stack, so the Grad-CAM target layer emits (channel x time) -- 512 channels, no
frequency axis anywhere in the forward graph. Attribution taken there localises
*when* the evidence is, not *at which frequency*.

So this module answers the temporal question honestly and cheaply, and
`tf_occlusion` (level 2b) answers the frequency question by a method that can
actually support it. Rendering this relevance over a separately computed mel
spectrogram is fine for a figure, as long as the caption says the highlighting
is temporal.

The method is gradient x activation summed over channels -- Grad-CAM's
formulation for a 1-D layer. It is implemented directly rather than through
captum because the model's forward takes (clips, mask) rather than a single
tensor, and wrapping it obscures more than the fifteen lines it would save.
"""

from __future__ import annotations

import numpy as np
import torch

from ..models.backbone import MERT_FRAME_RATE
from ..models.detector import Detector


def temporal_relevance(
    model: Detector,
    wav: torch.Tensor,
    clip_mask: torch.Tensor | None = None,
    normalize: bool = True,
) -> np.ndarray:
    """Relevance per conv frame for one song.

    Args:
        wav: [1, C, N] -- a single song's clips.
        clip_mask: [1, C] marking real clips.

    Returns:
        [C, T_conv] non-negative relevance. Positive values are evidence
        *towards the fake class*, since attribution is taken on the binary
        logit, which is signed towards fake.
    """
    if wav.dim() != 3 or wav.shape[0] != 1:
        raise ValueError(f"explain one song at a time: expected [1, C, N], got {tuple(wav.shape)}")

    # Every backbone parameter is frozen, so nothing inside the conv stack
    # creates a graph on its own -- the activations only become differentiable
    # if the *input* requires grad. Without this the hook captures a tensor
    # with requires_grad=False and no attribution is possible.
    wav = wav.detach().clone().requires_grad_(True)

    activations: list[torch.Tensor] = []

    def hook(_module, _inputs, output):
        output.retain_grad()
        activations.append(output)

    handle = model.backbone.conv_module.register_forward_hook(hook)
    try:
        was_training = model.training
        model.eval()
        # Gradients must reach inside the frozen backbone, so the training
        # fast path (no_grad) is explicitly disabled here.
        out = model.forward(wav, clip_mask=clip_mask, backbone_no_grad=False)
        model.zero_grad(set_to_none=True)
        out.binary_logit.sum().backward()
    finally:
        handle.remove()
        model.train(was_training)
        # Explaining must not contaminate training: the backward above leaves
        # gradients on the trainable head, which would otherwise be added to
        # the next optimiser step if an explanation is run mid-epoch.
        model.zero_grad(set_to_none=True)

    if not activations:
        raise RuntimeError("conv activations were not captured; check conv_module")

    act = activations[0]  # [B*C, channels, T_conv]
    if act.grad is None:
        raise RuntimeError("no gradient on conv activations; was the graph detached?")

    relevance = torch.relu((act * act.grad).sum(dim=1))  # [B*C, T_conv]
    relevance = relevance.detach().cpu().numpy()

    if normalize:
        peak = relevance.max(axis=-1, keepdims=True)
        relevance = np.divide(relevance, peak, out=np.zeros_like(relevance), where=peak > 0)
    return relevance


def relevance_times(n_frames: int, frame_rate: float = MERT_FRAME_RATE) -> np.ndarray:
    """Frame index -> seconds within the clip, for plotting and reporting."""
    return np.arange(n_frames) / frame_rate


def top_regions(
    relevance: np.ndarray, k: int = 5, frame_rate: float = MERT_FRAME_RATE
) -> list[tuple[float, float]]:
    """The k most relevant frames of one clip, as (start_s, end_s) intervals."""
    if relevance.ndim != 1:
        raise ValueError("pass a single clip's relevance vector")
    idx = np.argsort(relevance)[::-1][:k]
    step = 1.0 / frame_rate
    return sorted((float(i * step), float((i + 1) * step)) for i in idx)
