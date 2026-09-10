"""Classifier heads.

The diagram's single 4-way head (Real / Full / Half / Mostly Fake) is a
SONICS-specific taxonomy, so it cannot be evaluated on FakeMusicCaps -- which
is where the project's headline cross-generator number has to come from. The
split here keeps the taxonomy without letting it block the main experiment:

* `BinaryHead` is the primary output. It is what gets compared against the base
  paper's F1 0.99 -> 0.629 collapse, and it transfers across datasets.
* `TaxonomyHead` is auxiliary, trained on SONICS only.
* `SourceHead` covers the diagram's "Source & Label Prediction" box, scoped to
  FakeMusicCaps closed-set attribution -- SONICS only labels Suno vs Udio,
  which would not transfer.
"""

from __future__ import annotations

import torch
import torch.nn as nn

# SONICS taxonomy; index 0 is the only non-fake class, which is how the
# auxiliary head collapses to the binary label.
TAXONOMY_CLASSES = ("real", "full_fake", "half_fake", "mostly_fake")


def _mlp(dim: int, hidden: int, out: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, hidden),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden, out),
    )


class BinaryHead(nn.Module):
    """Real vs fake. Returns a single logit per example."""

    def __init__(self, dim: int = 768, hidden: int = 256, dropout: float = 0.1):
        super().__init__()
        self.net = _mlp(dim, hidden, 1, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class TaxonomyHead(nn.Module):
    """Auxiliary 4-way SONICS taxonomy."""

    def __init__(self, dim: int = 768, hidden: int = 256, dropout: float = 0.1):
        super().__init__()
        self.net = _mlp(dim, hidden, len(TAXONOMY_CLASSES), dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SourceHead(nn.Module):
    """Closed-set generator attribution (FakeMusicCaps)."""

    def __init__(self, dim: int = 768, n_sources: int = 5, hidden: int = 256, dropout: float = 0.1):
        super().__init__()
        self.net = _mlp(dim, hidden, n_sources, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
