"""Loss functions for training deep rPPG models.

Two complementary objectives used across the rPPG literature:

* ``NegPearsonLoss`` - a time-domain shape loss. Maximising the Pearson
  correlation between the predicted and reference BVP waveforms makes the
  network reproduce pulse morphology while staying invariant to the arbitrary
  scale/offset of the recovered signal (Yu et al., *rPPGNet*, 2019).
* ``FrequencyMSELoss`` - a frequency-domain loss. Matching the normalised power
  spectra pushes the dominant predicted frequency (the heart rate) towards the
  reference, which stabilises training when the waveform phase is hard to learn.

Both operate on batched 1-D signals of shape ``(B, T)`` and return a scalar.
Pure PyTorch, CPU-friendly, no I/O.
"""
from __future__ import annotations

import torch
import torch.nn as nn

_EPS = 1e-8


def _pearson_per_sample(pred: torch.Tensor, label: torch.Tensor,
                        eps: float = _EPS) -> torch.Tensor:
    """Return the Pearson correlation of each row pair.

    Args:
        pred: Predicted signals, shape ``(B, T)``.
        label: Reference signals, shape ``(B, T)``.
        eps: Small constant guarding the denominator (per-sample std product).

    Returns:
        Tensor of shape ``(B,)`` with the correlation of ``pred[i]`` vs
        ``label[i]`` in ``[-1, 1]``.
    """
    pred_c = pred - pred.mean(dim=1, keepdim=True)
    label_c = label - label.mean(dim=1, keepdim=True)
    covariance = (pred_c * label_c).sum(dim=1)
    pred_norm = torch.sqrt((pred_c * pred_c).sum(dim=1) + eps)
    label_norm = torch.sqrt((label_c * label_c).sum(dim=1) + eps)
    return covariance / (pred_norm * label_norm + eps)


class NegPearsonLoss(nn.Module):
    """1 minus the Pearson correlation, averaged over the batch.

    The loss is ``0`` for perfectly correlated signals and ``2`` for perfectly
    anti-correlated ones, so minimising it drives the predicted waveform to
    track the reference shape regardless of amplitude or offset.
    """

    def __init__(self, eps: float = _EPS) -> None:
        """Initialise the loss.

        Args:
            eps: Numerical floor added to the per-sample standard deviations.
        """
        super().__init__()
        self.eps = float(eps)

    def forward(self, pred: torch.Tensor,
                label: torch.Tensor) -> torch.Tensor:
        """Compute the mean negative-Pearson loss.

        Args:
            pred: Predicted BVP, shape ``(B, T)``.
            label: Reference BVP, shape ``(B, T)``.

        Returns:
            Scalar loss tensor ``mean(1 - pearson(pred_i, label_i))``.
        """
        if pred.shape != label.shape:
            raise ValueError(
                f"pred and label must match, got {tuple(pred.shape)} "
                f"vs {tuple(label.shape)}"
            )
        if pred.dim() != 2:
            raise ValueError(f"expected 2-D (B, T), got {pred.dim()}-D")
        correlation = _pearson_per_sample(pred, label, self.eps)
        return (1.0 - correlation).mean()


class FrequencyMSELoss(nn.Module):
    """MSE between the normalised power spectra of ``pred`` and ``label``.

    Each signal is transformed with a real FFT over the time axis; its power
    spectrum (``|rfft|^2``) is normalised to sum to one so the comparison is
    invariant to signal energy. The mean squared error between the two
    distributions penalises heart-rate (dominant-frequency) mismatch.
    """

    def __init__(self, eps: float = _EPS) -> None:
        """Initialise the loss.

        Args:
            eps: Numerical floor added to each spectrum's normalising total.
        """
        super().__init__()
        self.eps = float(eps)

    def forward(self, pred: torch.Tensor,
                label: torch.Tensor) -> torch.Tensor:
        """Compute the spectral MSE loss.

        Args:
            pred: Predicted BVP, shape ``(B, T)``.
            label: Reference BVP, shape ``(B, T)``.

        Returns:
            Scalar loss tensor: MSE between the sum-normalised power spectra.
        """
        if pred.shape != label.shape:
            raise ValueError(
                f"pred and label must match, got {tuple(pred.shape)} "
                f"vs {tuple(label.shape)}"
            )
        if pred.dim() != 2:
            raise ValueError(f"expected 2-D (B, T), got {pred.dim()}-D")
        pred_power = self._normalized_power(pred)
        label_power = self._normalized_power(label)
        return ((pred_power - label_power) ** 2).mean()

    def _normalized_power(self, signal: torch.Tensor) -> torch.Tensor:
        """Return the sum-normalised power spectrum of each row.

        Args:
            signal: Batched signals, shape ``(B, T)``.

        Returns:
            Tensor of shape ``(B, T // 2 + 1)`` whose rows sum to one.
        """
        centered = signal - signal.mean(dim=1, keepdim=True)
        spectrum = torch.fft.rfft(centered, dim=1)
        power = spectrum.real ** 2 + spectrum.imag ** 2
        total = power.sum(dim=1, keepdim=True) + self.eps
        return power / total
