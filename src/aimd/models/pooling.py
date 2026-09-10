"""Aggregation stages between the frozen MERT backbone and the classifier heads.

The architecture diagram drew four consecutive aggregation blocks -- "Layer
Attention", "Temporal Attention", "Frame Attention", "Self-Attention
Aggregation" and "Feature Fusion". Temporal and Frame attention operate on the
same axis (MERT frames *are* time frames) and Feature Fusion had a single
input, so the default path here collapses them to the axes that actually exist:

    [B, 13, T, D]  --LayerAttention-->  [B, T, D]  --TemporalAttention-->  [B, D]

and, one level up, [B, C, D] --SongAttention--> [B, D] over the clips of a song.
The dropped variants are kept as ablation options rather than deleted, so the
choices can be justified empirically instead of by assertion.

Every module returns its attention weights alongside its output: those weights
are Level 1 of the XAI layer, so they are part of the contract, not a debug aid.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def masked_softmax(scores: torch.Tensor, mask: torch.Tensor | None, dim: int) -> torch.Tensor:
    """Softmax that ignores masked positions. `mask` is True where valid."""
    if mask is not None:
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
    return scores.softmax(dim=dim)


class LayerNormStack(nn.Module):
    """Per-layer LayerNorm (+ optional projection) over MERT's hidden states.

    This is the concrete meaning of the diagram's "Layer-wise Feature
    Alignment" box. The 13 states are already [T, 768] so nothing needs
    geometric alignment, but their norms differ substantially across depth --
    without normalising first, the LayerAttention softmax is dominated by
    whichever layer happens to have the largest scale rather than by which
    layer is informative.
    """

    def __init__(self, n_layers: int = 13, dim: int = 768, proj_dim: int | None = None):
        super().__init__()
        self.norms = nn.ModuleList(nn.LayerNorm(dim) for _ in range(n_layers))
        self.proj = nn.Linear(dim, proj_dim) if proj_dim is not None else None
        self.out_dim = proj_dim if proj_dim is not None else dim

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """hidden: [B, L, T, D] -> [B, L, T, out_dim]"""
        normed = torch.stack([norm(hidden[:, i]) for i, norm in enumerate(self.norms)], dim=1)
        return self.proj(normed) if self.proj is not None else normed


class LayerAttention(nn.Module):
    """Weighted aggregation across MERT's 13 hidden states.

    `static` learns one scalar per layer (the SUPERB weighted-sum recipe): the
    weights are a global statement about which depths the detector relies on,
    which is what makes them readable as an explanation. `conditioned` scores
    layers per example, which is more expressive but yields a per-example
    weight vector instead of a single interpretable profile.

    Ablation: mode="last" bypasses aggregation and takes the final layer only,
    reproducing the common last-layer-features baseline.
    """

    def __init__(self, n_layers: int = 13, dim: int = 768, mode: str = "static"):
        super().__init__()
        if mode not in {"static", "conditioned", "last"}:
            raise ValueError(f"unknown layer attention mode: {mode!r}")
        self.mode = mode
        self.n_layers = n_layers
        if mode == "static":
            self.logits = nn.Parameter(torch.zeros(n_layers))
        elif mode == "conditioned":
            self.score = nn.Linear(dim, 1)

    def forward(
        self, hidden: torch.Tensor, layer_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """hidden: [B, L, T, D] -> ([B, T, D], weights)

        weights is [L] for static and [B, L] for conditioned; for "last" it is
        a one-hot [L] so downstream explanation code needs no special case.

        `layer_mask` ([L] bool) restricts aggregation to a subset of layers,
        renormalising the weights over what remains. This is the coalition
        operation SHAP needs (xai level 3): the value of a set of layers is the
        model's output when only those layers may contribute.
        """
        b, n_layers, _, _ = hidden.shape
        if self.mode == "last":
            w = F.one_hot(torch.tensor(n_layers - 1, device=hidden.device), n_layers).float()
            if layer_mask is not None:
                w = w * layer_mask.to(w.dtype)
            return torch.einsum("l,bltd->btd", w, hidden), w
        if self.mode == "static":
            w = masked_softmax(self.logits, layer_mask, dim=0)
            return torch.einsum("l,bltd->btd", w, hidden), w
        # conditioned: score each layer from its time-averaged representation
        scores = self.score(hidden.mean(dim=2)).squeeze(-1)  # [B, L]
        mask = layer_mask.unsqueeze(0).expand_as(scores) if layer_mask is not None else None
        w = masked_softmax(scores, mask, dim=-1)
        return torch.einsum("bl,bltd->btd", w, hidden), w


class AttentivePool(nn.Module):
    """Additive attention pooling over one axis, with a mean-pool ablation.

    Used for both the frame axis (T) and the clip axis (C) -- the mechanism is
    identical, only the axis differs, which is precisely why the diagram's
    separate "Temporal" and "Frame" blocks were redundant.
    """

    def __init__(self, dim: int = 768, hidden: int = 128, mode: str = "attn"):
        super().__init__()
        if mode not in {"attn", "mean"}:
            raise ValueError(f"unknown pooling mode: {mode!r}")
        self.mode = mode
        if mode == "attn":
            self.score = nn.Sequential(nn.Linear(dim, hidden), nn.Tanh(), nn.Linear(hidden, 1))

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """x: [B, N, D], mask: [B, N] True where valid -> ([B, D], [B, N])"""
        if self.mode == "mean":
            if mask is None:
                w = x.new_full(x.shape[:2], 1.0 / x.shape[1])
            else:
                w = mask.float() / mask.float().sum(dim=1, keepdim=True).clamp(min=1.0)
            return torch.einsum("bn,bnd->bd", w, x), w
        scores = self.score(x).squeeze(-1)  # [B, N]
        w = masked_softmax(scores, mask, dim=-1)
        return torch.einsum("bn,bnd->bd", w, x), w


class SelfAttentionBlock(nn.Module):
    """Optional pre-pooling self-attention over frames (diagram ablation).

    Off by default: it sits between two aggregation stages that already mix
    across time, so it needs an ablation to earn its parameters.
    """

    def __init__(self, dim: int = 768, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        pad = ~mask if mask is not None else None
        out, _ = self.attn(x, x, x, key_padding_mask=pad, need_weights=False)
        return self.norm(x + out)
