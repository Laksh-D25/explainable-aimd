"""Test fixtures.

The tests deliberately run without downloading MERT: the checkpoint is ~400 MB
and the shape/gradient contracts under test belong to the trainable tail, not
to the backbone weights. `StubBackbone` mirrors MertBackbone's interface --
including frozen parameters that still pass gradients through -- so the same
tests exercise the real assembly. Tests that genuinely need the real weights
are marked `real_mert` and skipped unless it is cached locally.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

SAMPLE_RATE = 24_000
FRAME_RATE = 75


class StubBackbone(nn.Module):
    """Shape-compatible stand-in for MertBackbone with a real gradient path."""

    def __init__(self, n_layers: int = 13, dim: int = 32, sample_rate: int = SAMPLE_RATE):
        super().__init__()
        self.n_layers = n_layers
        self.dim = dim
        self.sample_rate = sample_rate
        self.max_seconds = 30.0
        # A stride-320 conv stack mimics MERT's 24 kHz -> 75 Hz reduction.
        self.conv = nn.Conv1d(1, dim, kernel_size=320, stride=320)
        self.layer_mix = nn.ModuleList(nn.Linear(dim, dim) for _ in range(n_layers))
        self.requires_grad_(False)  # frozen, like the real backbone

    @property
    def conv_module(self) -> nn.Module:
        return self.conv

    def train(self, mode: bool = True):
        # Mirrors MertBackbone: the wrapper follows the mode, the weights stay in eval.
        super().train(mode)
        self.conv.eval()
        self.layer_mix.eval()
        return self

    def forward(self, wav: torch.Tensor, no_grad: bool = False) -> torch.Tensor:
        ctx = torch.no_grad() if no_grad else torch.enable_grad()
        with ctx:
            feats = self.conv(wav.unsqueeze(1)).transpose(1, 2)  # [B, T, D]
            return torch.stack([m(feats) for m in self.layer_mix], dim=1)  # [B, L, T, D]


@pytest.fixture
def stub_backbone() -> StubBackbone:
    torch.manual_seed(0)
    return StubBackbone()


@pytest.fixture
def noise():
    """Synthetic audio: [B, C, N] clips of `seconds` at 24 kHz."""

    def _make(batch: int = 2, clips: int = 3, seconds: float = 1.0) -> torch.Tensor:
        torch.manual_seed(1)
        return torch.randn(batch, clips, int(seconds * SAMPLE_RATE))

    return _make
