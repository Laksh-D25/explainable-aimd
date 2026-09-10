"""The assembled detector: frozen MERT -> layer attn -> temporal pool -> song pool -> heads.

Song-level aggregation is the stage the architecture diagram was missing. SONICS
songs run 32-240 s while MERT is a 5 s model, so audio must be segmented into
clips; without an explicit clip->song stage the model emits per-clip decisions
while SONICS's ~0.97 F1 baseline and the base paper's numbers are per-song, and
the results are not comparable to either.

`forward_from_hidden` is split out deliberately: the frozen backbone is by far
the expensive part, so SHAP over the 13 layer contributions (XAI level 3) and
the ablation sweeps re-run only the trainable tail against cached hidden states.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .backbone import MertBackbone
from .heads import BinaryHead, SourceHead, TaxonomyHead
from .pooling import AttentivePool, LayerAttention, LayerNormStack, SelfAttentionBlock


@dataclass
class DetectorOutput:
    """Predictions plus the attention weights that form XAI level 1."""

    binary_logit: torch.Tensor  # [B]
    embedding: torch.Tensor  # [B, D] song-level
    layer_weights: torch.Tensor  # [L] (static) or [B*C, L] (conditioned)
    frame_weights: torch.Tensor  # [B*C, T]
    clip_weights: torch.Tensor  # [B, C]
    taxonomy_logits: torch.Tensor | None = None  # [B, 4]
    source_logits: torch.Tensor | None = None  # [B, n_sources]


class Detector(nn.Module):
    def __init__(
        self,
        backbone: MertBackbone | None = None,
        *,
        model_name: str = "m-a-p/MERT-v1-95M",
        proj_dim: int | None = None,
        layer_mode: str = "static",  # ablation: static | conditioned | last
        frame_pool: str = "attn",  # ablation: attn | mean
        song_pool: str = "attn",  # ablation: attn | mean
        use_self_attn: bool = False,  # ablation: the diagram's self-attention block
        head_hidden: int = 256,
        dropout: float = 0.1,
        use_taxonomy_head: bool = True,
        n_sources: int | None = None,
    ):
        super().__init__()
        self.backbone = backbone if backbone is not None else MertBackbone(model_name=model_name)

        self.align = LayerNormStack(self.backbone.n_layers, self.backbone.dim, proj_dim)
        dim = self.align.out_dim

        self.layer_attn = LayerAttention(self.backbone.n_layers, dim, mode=layer_mode)
        self.self_attn = SelfAttentionBlock(dim, dropout=dropout) if use_self_attn else None
        self.frame_pool = AttentivePool(dim, mode=frame_pool)
        self.song_pool = AttentivePool(dim, mode=song_pool)

        self.binary_head = BinaryHead(dim, head_hidden, dropout)
        self.taxonomy_head = TaxonomyHead(dim, head_hidden, dropout) if use_taxonomy_head else None
        self.source_head = SourceHead(dim, n_sources, head_hidden, dropout) if n_sources else None

    def trainable_parameters(self):
        return (p for p in self.parameters() if p.requires_grad)

    def forward_from_hidden(
        self,
        hidden: torch.Tensor,
        clip_mask: torch.Tensor | None = None,
        frame_mask: torch.Tensor | None = None,
        layer_mask: torch.Tensor | None = None,
    ) -> DetectorOutput:
        """hidden: [B, C, L, T, D] -> DetectorOutput.

        Kept separate from `forward` so callers that already hold backbone
        features (SHAP, ablation sweeps, the cached dev subset) never pay for
        MERT again.
        """
        b, c = hidden.shape[:2]
        flat = hidden.flatten(0, 1)  # [B*C, L, T, D]

        aligned = self.align(flat)
        x, layer_w = self.layer_attn(aligned, layer_mask)  # [B*C, T, D]

        fmask = frame_mask.flatten(0, 1) if frame_mask is not None else None
        if self.self_attn is not None:
            x = self.self_attn(x, fmask)

        clip_emb, frame_w = self.frame_pool(x, fmask)  # [B*C, D]
        song_emb, clip_w = self.song_pool(clip_emb.view(b, c, -1), clip_mask)  # [B, D]

        return DetectorOutput(
            binary_logit=self.binary_head(song_emb),
            embedding=song_emb,
            layer_weights=layer_w,
            frame_weights=frame_w,
            clip_weights=clip_w,
            taxonomy_logits=self.taxonomy_head(song_emb) if self.taxonomy_head else None,
            source_logits=self.source_head(song_emb) if self.source_head else None,
        )

    def forward(
        self,
        wav: torch.Tensor,
        clip_mask: torch.Tensor | None = None,
        frame_mask: torch.Tensor | None = None,
        backbone_no_grad: bool = True,
    ) -> DetectorOutput:
        """wav: [B, C, N] clips of a song at 24 kHz (use C=1 for clip-level training).

        `backbone_no_grad` is True on the training path (the backbone is frozen,
        so its activations need no graph) and must be set False whenever an
        explanation needs gradients to reach the audio.
        """
        if wav.dim() == 2:  # [B, N] -> single clip per song
            wav = wav.unsqueeze(1)
        b, c, n = wav.shape
        flat = wav.reshape(b * c, n)

        if clip_mask is None:
            hidden = self.backbone(flat, no_grad=backbone_no_grad)
        else:
            # Padded clips contribute nothing -- the song pool gives them zero
            # weight -- so running the backbone over them is pure waste. Songs
            # vary in length, so on a real batch this is a meaningful saving.
            valid = clip_mask.reshape(-1)
            packed = self.backbone(flat[valid], no_grad=backbone_no_grad)
            hidden = packed.new_zeros((b * c, *packed.shape[1:]))
            hidden[valid] = packed

        hidden = hidden.view(b, c, *hidden.shape[1:])
        return self.forward_from_hidden(hidden, clip_mask, frame_mask)
