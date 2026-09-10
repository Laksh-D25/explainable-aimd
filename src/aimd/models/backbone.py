"""Frozen MERT feature extractor.

Note on the architecture diagram: the "RVQ-VAE Acoustic Teacher" and "CQT
Musical Teacher" boxes drawn inside this stage are MERT's *pre-training*
masked-prediction targets. The released checkpoint is a 1-D convolutional
feature extractor plus a 12-layer transformer and contains no teachers, so
there is nothing to implement here -- the teachers explain where the weights
came from, not what runs at inference.

Two consequences drive this module's design:

* MERT was pre-trained on 5 s clips at a 75 Hz frame rate and its attention is
  O(T^2). A 120 s input is ~9000 frames, whose attention alone is ~1.9 GB per
  layer in fp16. Long audio must be segmented upstream; `max_seconds` guards
  against accidentally feeding a whole song.
* "Frozen" means the parameters do not update, NOT that gradients cannot flow.
  Grad-CAM and the faithfulness metrics need a backward pass reaching the
  input, so freezing is done with `requires_grad_(False)` on parameters and
  `no_grad` is applied only in the explicit feature-extraction fast path.
"""

from __future__ import annotations

import warnings

import torch
import torch.nn as nn

MERT_SAMPLE_RATE = 24_000
MERT_FRAME_RATE = 75  # hidden-state frames per second


class MertBackbone(nn.Module):
    def __init__(
        self,
        model_name: str = "m-a-p/MERT-v1-95M",
        max_seconds: float = 30.0,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        from transformers import AutoConfig, AutoModel, Wav2Vec2FeatureExtractor

        self.model_name = model_name
        self.max_seconds = max_seconds

        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            model_name, trust_remote_code=True, torch_dtype=dtype
        )
        self.model.eval()
        self.model.requires_grad_(False)  # freeze weights; gradients still flow

        # Read normalisation behaviour from the shipped preprocessor rather than
        # assuming it, then apply it in torch -- the HF processor is numpy and
        # would dominate the data loader.
        fe = Wav2Vec2FeatureExtractor.from_pretrained(model_name, trust_remote_code=True)
        self.do_normalize = bool(getattr(fe, "do_normalize", True))
        self.sample_rate = int(getattr(fe, "sampling_rate", MERT_SAMPLE_RATE))

        self.dim = int(config.hidden_size)
        self.n_layers = int(config.num_hidden_layers) + 1  # + the embedding output

    @property
    def conv_module(self) -> nn.Module:
        """The 1-D conv stack, i.e. the Grad-CAM target layer.

        Its output is (time x channel), not (time x frequency) -- MERT never
        builds a spectrogram -- so attributions taken here localise *when*, not
        *at which frequency*. See aimd.xai.relevance (L2a) and
        aimd.xai.tf_occlusion (L2b) for the two halves of Level 2.
        """
        return self.model.feature_extractor

    def train(self, mode: bool = True):  # noqa: D102 - keep the backbone in eval
        super().train(mode)
        self.model.eval()
        return self

    def normalize(self, wav: torch.Tensor) -> torch.Tensor:
        if not self.do_normalize:
            return wav
        mean = wav.mean(dim=-1, keepdim=True)
        std = wav.std(dim=-1, keepdim=True)
        return (wav - mean) / (std + 1e-7)

    def forward(self, wav: torch.Tensor, no_grad: bool = False) -> torch.Tensor:
        """wav: [B, N] mono at 24 kHz -> hidden states [B, n_layers, T, dim].

        Set `no_grad=True` for the training/eval fast path. Leave it False when
        the caller needs gradients through the backbone (XAI).
        """
        if wav.dim() != 2:
            raise ValueError(f"expected [B, N] mono waveform, got shape {tuple(wav.shape)}")
        seconds = wav.shape[-1] / self.sample_rate
        if seconds > self.max_seconds:
            warnings.warn(
                f"{seconds:.1f}s input exceeds max_seconds={self.max_seconds}. MERT attention is "
                "O(T^2) at 75 Hz; segment into clips upstream instead.",
                stacklevel=2,
            )
        wav = self.normalize(wav.to(next(self.model.parameters()).dtype))
        ctx = torch.no_grad() if no_grad else torch.enable_grad()
        with ctx:
            out = self.model(wav, output_hidden_states=True)
        return torch.stack(out.hidden_states, dim=1)
