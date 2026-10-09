"""TS-CAN: Temporal Shift Convolutional Attention Network for rPPG.

A simplified but faithful implementation of TS-CAN (Liu, Fromm, McDuff,
"Multi-Task Temporal Shift Attention Networks for On-Device Contactless
Vitals Measurement", NeurIPS 2020).

Two 2D-CNN branches process every frame (time folded into the batch):

  * a *motion* branch fed normalized consecutive frame differences, and
  * an *appearance* branch fed the standardized frames.

The appearance branch emits a sigmoid spatial attention mask at two depths.
Each mask gates the motion branch's feature maps, so the network learns where
on the face the pulse signal lives. A global pool and a linear head turn each
frame's gated features into one scalar; stacking them gives the BVP waveform.

Uniform interface: ``forward(x)`` takes ``x`` of shape ``(B, 3, T, H, W)`` and
returns the BVP of shape ``(B, T)``. Adaptive pooling makes any spatial size
(including H=W=72) work on CPU. No I/O, no global RNG.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

_EPS: float = 1e-7


def _per_sample_zscore(x: torch.Tensor) -> torch.Tensor:
    """Z-score each clip over its channel/time/spatial dims.

    Args:
        x: Tensor of shape ``(B, C, T, H, W)``.

    Returns:
        Tensor of the same shape, zero-mean/unit-variance per batch element.
    """
    dims = (1, 2, 3, 4)
    mean = x.mean(dim=dims, keepdim=True)
    std = x.std(dim=dims, keepdim=True)
    return (x - mean) / (std + _EPS)


def _diff_normalized(x: torch.Tensor) -> torch.Tensor:
    """Build the DiffNormalized motion stream, padded back to length ``T``.

    ``d[t] = (f[t+1] - f[t]) / (f[t+1] + f[t] + eps)`` for the ``T - 1``
    available differences; one zero frame is appended so the stream keeps the
    original temporal length. The result is then z-scored per clip.

    Args:
        x: Raw frames of shape ``(B, 3, T, H, W)``.

    Returns:
        Motion stream of shape ``(B, 3, T, H, W)``.
    """
    nxt, cur = x[:, :, 1:], x[:, :, :-1]
    diff = (nxt - cur) / (nxt + cur + _EPS)
    pad = torch.zeros_like(diff[:, :, :1])
    diff = torch.cat([diff, pad], dim=2)
    return _per_sample_zscore(diff)


def _normalize_mask(mask: torch.Tensor) -> torch.Tensor:
    """Spatially normalize a soft attention mask (TS-CAN softmax-style scaling).

    Scales each ``(1, H, W)`` mask so its values sum to ``H * W / 2``, keeping
    the gating energy stable regardless of spatial resolution.

    Args:
        mask: Sigmoid attention map of shape ``(N, 1, H, W)``.

    Returns:
        Normalized mask of the same shape.
    """
    _, _, h, w = mask.shape
    denom = mask.sum(dim=(2, 3), keepdim=True) * 2.0 + _EPS
    return mask / denom * float(h * w)


class _ConvBlock(nn.Module):
    """Two padded 3x3 convolutions, each followed by BN and ReLU."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the two-conv block.

        Args:
            x: Tensor of shape ``(N, in_ch, H, W)``.

        Returns:
            Tensor of shape ``(N, out_ch, H, W)``.
        """
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return x


class TSCAN(nn.Module):
    """TS-CAN attention network mapping a face clip to a BVP waveform."""

    def __init__(
        self,
        base_channels: int = 32,
        dropout: float = 0.25,
        fc_hidden: int = 128,
    ) -> None:
        """Build the two-branch attention network.

        Args:
            base_channels: Channels in the first conv stage (doubled in stage 2).
            dropout: Dropout probability after each pooling step.
            fc_hidden: Width of the hidden linear layer in the regression head.
        """
        super().__init__()
        c1 = base_channels
        c2 = base_channels * 2

        # Stage 1 (c1 channels) and stage 2 (c2 channels) for both branches.
        self.app_block1 = _ConvBlock(3, c1)
        self.mot_block1 = _ConvBlock(3, c1)
        self.app_block2 = _ConvBlock(c1, c2)
        self.mot_block2 = _ConvBlock(c1, c2)

        # Appearance-driven spatial attention masks.
        self.att1 = nn.Conv2d(c1, 1, kernel_size=1)
        self.att2 = nn.Conv2d(c2, 1, kernel_size=1)

        self.pool = nn.AvgPool2d(kernel_size=2)
        self.drop = nn.Dropout(p=dropout)
        self.gpool = nn.AdaptiveAvgPool2d(1)

        # Per-frame regression head on the gated motion features.
        self.fc1 = nn.Linear(c2, fc_hidden)
        self.fc2 = nn.Linear(fc_hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Estimate the BVP waveform for a batch of face clips.

        Args:
            x: Float frames of shape ``(B, 3, T, H, W)``.

        Returns:
            Per-frame BVP of shape ``(B, T)``.
        """
        if x.dim() != 5 or x.size(1) != 3:
            raise ValueError(f"expected (B, 3, T, H, W), got {tuple(x.shape)}")
        b, _, t, h, w = x.shape

        motion = _diff_normalized(x)
        appearance = _per_sample_zscore(x)

        # Fold time into the batch dimension -> (B*T, 3, H, W).
        mot = motion.permute(0, 2, 1, 3, 4).reshape(b * t, 3, h, w)
        app = appearance.permute(0, 2, 1, 3, 4).reshape(b * t, 3, h, w)

        # Stage 1: extract features, gate motion by appearance attention.
        app = self.app_block1(app)
        mot = self.mot_block1(mot)
        mot = mot * _normalize_mask(torch.sigmoid(self.att1(app)))
        app = self.drop(self.pool(app))
        mot = self.drop(self.pool(mot))

        # Stage 2: deeper features, second attention gate.
        app = self.app_block2(app)
        mot = self.mot_block2(mot)
        mot = mot * _normalize_mask(torch.sigmoid(self.att2(app)))
        mot = self.drop(self.pool(mot))

        # Global pool -> per-frame scalar -> reshape to (B, T).
        feat = self.gpool(mot).flatten(1)
        feat = F.relu(self.fc1(feat))
        out = self.fc2(feat)
        return out.view(b, t)
