"""PhysNet 3D-CNN for remote photoplethysmography (rPPG).

Reference
---------
Zitong Yu, Xiaobai Li, Guoying Zhao. "Remote Photoplethysmograph Signal
Measurement from Facial Videos Using Spatio-Temporal Networks." BMVC 2019.

The network is a spatio-temporal 3-D convolutional encoder-decoder. The
encoder progressively pools the spatial dimensions (and mildly pools time)
to distil a temporal feature trace; the decoder upsamples time back toward
the input length, and a final 1x1x1 convolution plus spatial global pooling
yields one blood-volume-pulse (BVP) value per frame.

Uniform model interface
------------------------
``forward(x)`` takes ``x`` of shape ``(B, 3, T, H, W)`` and returns a BVP
tensor of shape ``(B, T)``. Adaptive pooling makes the model agnostic to the
spatial size (``H == W == 72`` works) and the final adaptive temporal pool
guarantees the output length equals the input ``T`` for arbitrary ``T``.
"""
from __future__ import annotations

import torch
import torch.nn as nn

# Channel widths for the encoder/decoder stages (kept small for CPU use).
_C1 = 16
_C2 = 32
_C3 = 64

# Temporal up-sampling block geometry: kernel 4, stride 2, padding 1 doubles
# the temporal length (out = (in - 1) * 2 - 2 + 4 = 2 * in).
_UP_KERNEL = (4, 1, 1)
_UP_STRIDE = (2, 1, 1)
_UP_PADDING = (1, 0, 0)


class _ConvBlock3d(nn.Module):
    """Conv3d -> BatchNorm3d -> ReLU that preserves spatial/temporal size.

    Args:
        in_ch: Number of input channels.
        out_ch: Number of output channels.
        kernel: 3-D convolution kernel (temporal, height, width).
        padding: Padding chosen so the block keeps the input resolution.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel: tuple[int, int, int],
        padding: tuple[int, int, int],
    ) -> None:
        super().__init__()
        self.conv = nn.Conv3d(in_ch, out_ch, kernel, stride=1, padding=padding)
        self.norm = nn.BatchNorm3d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply convolution, normalisation and activation."""
        return self.act(self.norm(self.conv(x)))


class _UpBlock3d(nn.Module):
    """ConvTranspose3d -> BatchNorm3d -> ELU that doubles the temporal length."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.deconv = nn.ConvTranspose3d(
            in_ch, out_ch, _UP_KERNEL, stride=_UP_STRIDE, padding=_UP_PADDING
        )
        self.norm = nn.BatchNorm3d(out_ch)
        self.act = nn.ELU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Upsample time by a factor of two."""
        return self.act(self.norm(self.deconv(x)))


class PhysNet(nn.Module):
    """PhysNet spatio-temporal encoder-decoder for BVP recovery.

    Args:
        in_ch: Number of input channels (3 for RGB streams).
    """

    def __init__(self, in_ch: int = 3) -> None:
        super().__init__()

        # Stem: wide spatial kernel, no temporal mixing, then spatial /2.
        # ``ceil_mode=True`` keeps odd/small dimensions from collapsing to zero,
        # so the model tolerates arbitrary ``T`` (e.g. T < 4) and spatial sizes.
        self.stem = _ConvBlock3d(in_ch, _C1, (1, 5, 5), (0, 2, 2))
        self.pool_spatial_1 = nn.MaxPool3d(
            (1, 2, 2), stride=(1, 2, 2), ceil_mode=True
        )

        # Encoder stage 1: full 3-D mixing, then spatio-temporal /2.
        self.enc1a = _ConvBlock3d(_C1, _C2, (3, 3, 3), (1, 1, 1))
        self.enc1b = _ConvBlock3d(_C2, _C3, (3, 3, 3), (1, 1, 1))
        self.pool_st_1 = nn.MaxPool3d((2, 2, 2), stride=(2, 2, 2), ceil_mode=True)

        # Encoder stage 2: 3-D mixing, then spatio-temporal /2.
        self.enc2a = _ConvBlock3d(_C3, _C3, (3, 3, 3), (1, 1, 1))
        self.enc2b = _ConvBlock3d(_C3, _C3, (3, 3, 3), (1, 1, 1))
        self.pool_st_2 = nn.MaxPool3d((2, 2, 2), stride=(2, 2, 2), ceil_mode=True)

        # Encoder stage 3: 3-D mixing, then spatial /2 (time preserved).
        self.enc3a = _ConvBlock3d(_C3, _C3, (3, 3, 3), (1, 1, 1))
        self.enc3b = _ConvBlock3d(_C3, _C3, (3, 3, 3), (1, 1, 1))
        self.pool_spatial_2 = nn.MaxPool3d(
            (1, 2, 2), stride=(1, 2, 2), ceil_mode=True
        )

        # Bottleneck: 3-D mixing at the coarsest spatial resolution.
        self.bottleneck_a = _ConvBlock3d(_C3, _C3, (3, 3, 3), (1, 1, 1))
        self.bottleneck_b = _ConvBlock3d(_C3, _C3, (3, 3, 3), (1, 1, 1))

        # Decoder: restore the temporal resolution (x2 then x2).
        self.up1 = _UpBlock3d(_C3, _C3)
        self.up2 = _UpBlock3d(_C3, _C3)

        # Head: 1x1x1 projection to a single channel + global spatial pooling.
        self.head = nn.Conv3d(_C3, 1, (1, 1, 1), stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map a video clip to a per-frame BVP signal.

        Args:
            x: FloatTensor of shape ``(B, 3, T, H, W)``.

        Returns:
            FloatTensor of shape ``(B, T)`` holding one BVP value per frame.

        Raises:
            ValueError: If ``x`` is not a 5-D tensor.
        """
        if x.dim() != 5:
            raise ValueError(
                f"PhysNet expects (B, C, T, H, W); got shape {tuple(x.shape)}"
            )

        num_frames = x.shape[2]

        feats = self.stem(x)
        feats = self.pool_spatial_1(feats)

        feats = self.enc1a(feats)
        feats = self.enc1b(feats)
        feats = self.pool_st_1(feats)

        feats = self.enc2a(feats)
        feats = self.enc2b(feats)
        feats = self.pool_st_2(feats)

        feats = self.enc3a(feats)
        feats = self.enc3b(feats)
        feats = self.pool_spatial_2(feats)

        feats = self.bottleneck_a(feats)
        feats = self.bottleneck_b(feats)

        feats = self.up1(feats)
        feats = self.up2(feats)

        # Project to one channel, then force the temporal length back to T and
        # collapse the spatial dimensions. AdaptiveAvgPool3d makes the result
        # robust to arbitrary T and spatial size.
        feats = self.head(feats)
        feats = nn.functional.adaptive_avg_pool3d(feats, (num_frames, 1, 1))

        # (B, 1, T, 1, 1) -> (B, T).
        return feats.reshape(feats.shape[0], num_frames)
