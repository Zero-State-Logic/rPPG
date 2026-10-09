"""ProPOSNet - a differentiable, learnable generalization of the ProPOS algorithm.

Our paper's core contribution. Classical ProPOS (Rao et al. 2025) rotates the POS
projection plane into a *fixed* bank of 50 bases, projects the RGB trace onto each,
and selects one candidate pulse by a hand-crafted SNR criterion. ProPOSNet makes all
three steps learnable and end-to-end differentiable:

  * the projection bank is an nn.Parameter **initialized from the classical ProPOS
    bank**, then fine-tuned by gradient descent;
  * skin-pixel aggregation uses a learned spatial-attention pool instead of a flat mean;
  * candidate selection becomes a soft, differentiable attention over the bank (a
    learnable stand-in for the SNR argmax), followed by a small temporal CNN that
    plays the role of ProPOS's sinusoidal signal-fitting step.

Interface: forward(x: (B, 3, T, H, W)) -> bvp (B, T).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from rppg.methods.classical.propos import build_projection_bank

_EPS = 1e-6


class ProPOSNet(nn.Module):
    """Learnable Projection of Rotated Orthogonal bases in POS."""

    def __init__(self, hidden: int = 64, t_kernel: int = 5) -> None:
        super().__init__()
        bank = build_projection_bank()                       # (K, 2, 3) classical init
        self.bank = nn.Parameter(torch.tensor(bank, dtype=torch.float32))
        self.n_bases = bank.shape[0]
        self.alpha = nn.Parameter(torch.ones(self.n_bases))  # per-basis combine weight

        self.spatial_attn = nn.Conv2d(3, 1, kernel_size=1)   # learned skin weighting
        self.scorer = nn.Sequential(                         # differentiable SNR-selection
            nn.Linear(3, 32), nn.ReLU(), nn.Linear(32, 1),
        )
        pad = t_kernel // 2
        self.temporal = nn.Sequential(                       # learned signal-fitting
            nn.Conv1d(1, hidden, t_kernel, padding=pad),
            nn.BatchNorm1d(hidden), nn.ReLU(),
            nn.Conv1d(hidden, hidden, t_kernel, padding=pad),
            nn.BatchNorm1d(hidden), nn.ReLU(),
            nn.Conv1d(hidden, 1, t_kernel, padding=pad),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, t, h, w = x.shape
        # Learned spatial-attention pooling -> per-frame RGB trace (B, 3, T).
        frames = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        attn = self.spatial_attn(frames).reshape(b * t, 1, h * w)
        attn = torch.softmax(attn, dim=-1).reshape(b * t, 1, h, w)
        pooled = (frames * attn).sum(dim=(2, 3))             # (B*T, 3)
        trace = pooled.reshape(b, t, c).permute(0, 2, 1)     # (B, 3, T)

        # Per-channel temporal normalization (ProPOS Eq. 2).
        mu = trace.mean(dim=2, keepdim=True)
        sd = trace.std(dim=2, keepdim=True) + _EPS
        tn = (trace - mu) / sd                               # (B, 3, T)

        # Project onto the (learnable) rotated bank -> candidate pulses (B, K, T).
        proj = torch.einsum("kij,bjt->bkit", self.bank, tn)  # (B, K, 2, T)
        s1, s2 = proj[:, :, 0, :], proj[:, :, 1, :]
        candidates = s1 + self.alpha.view(1, -1, 1) * s2      # (B, K, T)

        # Soft, differentiable candidate selection (learned SNR analog).
        feats = torch.stack(
            [candidates.std(dim=2),
             candidates.amax(dim=2) - candidates.amin(dim=2),
             candidates.abs().mean(dim=2)], dim=-1)           # (B, K, 3)
        weights = torch.softmax(self.scorer(feats).squeeze(-1), dim=1)  # (B, K)
        selected = (weights.unsqueeze(-1) * candidates).sum(dim=1)      # (B, T)

        # Temporal refinement (learned signal fitting).
        return self.temporal(selected.unsqueeze(1)).squeeze(1)          # (B, T)
