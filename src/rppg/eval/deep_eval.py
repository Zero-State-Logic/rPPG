"""Window-wise heart-rate evaluation of a trained deep rPPG model.

A trained model conforms to the uniform deep interface: ``forward(x)`` takes
``x`` of shape ``(B, 3, T, H, W)`` and returns a BVP tensor of shape
``(B, T)``. For every window in a loader this module estimates the predicted
heart rate from the model's BVP and the reference heart rate from the label
BVP, then aggregates HR error metrics across all windows.

Reuses the shared estimators in :mod:`rppg.hr.metrics` so deep and classical
pipelines are scored identically. No file writes or network access happen at
import time.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn

from rppg.hr.metrics import accuracy_within, estimate_hr_fft, mae, rmse

# Keys used by the project's collate shapes, mirrored from the trainer so the
# same loaders feed both training and evaluation.
_INPUT_KEYS = ("input", "inputs", "x", "frames")
_TARGET_KEYS = ("target", "targets", "y", "bvp", "label")

_WITHIN_TOL_BPM = 5.0
_EPS = 1e-9


def _resolve_device(device: Optional[Any]) -> torch.device:
    """Return an explicit device, defaulting to CUDA when available."""
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _unpack_batch(batch: Any) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract ``(inputs, targets)`` tensors from a batch.

    Supports the common collate shapes: a ``(inputs, targets, ...)`` tuple or
    list, or a mapping exposing the tensors under conventional keys.

    Args:
        batch: One item yielded by the loader.

    Returns:
        A ``(inputs, targets)`` pair of float tensors (left on their device).

    Raises:
        ValueError: If a sequence batch has fewer than two entries.
        TypeError: If the batch type is unsupported.
    """
    if isinstance(batch, dict):
        in_key = next((k for k in _INPUT_KEYS if k in batch), None)
        tgt_key = next((k for k in _TARGET_KEYS if k in batch), None)
        if in_key is None or tgt_key is None:
            raise ValueError("Batch dict is missing input/target keys.")
        inputs, targets = batch[in_key], batch[tgt_key]
    elif isinstance(batch, (tuple, list)):
        if len(batch) < 2:
            raise ValueError("Batch must provide at least (inputs, targets).")
        inputs, targets = batch[0], batch[1]
    else:
        raise TypeError(f"Unsupported batch type: {type(batch)!r}")

    inputs = torch.as_tensor(inputs, dtype=torch.float32)
    targets = torch.as_tensor(targets, dtype=torch.float32)
    return inputs, targets


def _as_2d(signal: np.ndarray) -> np.ndarray:
    """Return ``signal`` as a ``(B, T)`` array, promoting a bare ``(T,)`` row."""
    arr = np.asarray(signal, dtype=np.float64)
    if arr.ndim == 1:
        return arr[np.newaxis, :]
    return arr.reshape(arr.shape[0], -1)


def _pearson(pred: np.ndarray, true: np.ndarray) -> float:
    """Pearson correlation of two 1-D arrays.

    Returns ``nan`` when fewer than two points remain or either series is
    constant (zero variance), both of which leave the coefficient undefined.
    """
    pred = np.asarray(pred, dtype=np.float64)
    true = np.asarray(true, dtype=np.float64)
    if pred.size < 2 or pred.std() < _EPS or true.std() < _EPS:
        return float("nan")
    return float(np.corrcoef(pred, true)[0, 1])


def _window_hrs(
    pred_bvp: np.ndarray, true_bvp: np.ndarray, fps: float
) -> Tuple[List[float], List[float]]:
    """Estimate per-window predicted and reference HR for one batch.

    Args:
        pred_bvp: Model BVP, shape ``(B, T)`` or ``(T,)``.
        true_bvp: Label BVP, shape ``(B, T)`` or ``(T,)``.
        fps: Sampling rate of both signals in frames per second.

    Returns:
        ``(pred_hr, true_hr)`` lists, one bpm estimate per window.
    """
    pred_rows = _as_2d(pred_bvp)
    true_rows = _as_2d(true_bvp)
    count = min(pred_rows.shape[0], true_rows.shape[0])

    pred_hr: List[float] = []
    true_hr: List[float] = []
    for i in range(count):
        pred_hr.append(estimate_hr_fft(pred_rows[i], fps))
        true_hr.append(estimate_hr_fft(true_rows[i], fps))
    return pred_hr, true_hr


def _empty_result() -> Dict[str, Any]:
    """Return the result dict for the degenerate (no valid windows) case."""
    empty = np.empty(0, dtype=np.float64)
    return {
        "mae": float("nan"),
        "rmse": float("nan"),
        "within5": float("nan"),
        "pearson": float("nan"),
        "n": 0,
        "hr_pred": empty,
        "hr_true": empty,
    }


def evaluate_hr(
    model: nn.Module,
    loader: Any,
    fps: float,
    device: Optional[Any] = None,
) -> Dict[str, Any]:
    """Score a trained model's window-wise heart-rate accuracy.

    Runs ``model`` over every batch in ``loader`` (eval mode, no gradients),
    derives a predicted and a reference HR per window, drops any window whose
    estimate is non-finite, and aggregates the surviving pairs.

    Args:
        model: Module with the uniform ``(B, 3, T, H, W) -> (B, T)`` interface.
        loader: Iterable of batches yielding ``(inputs, targets)``.
        fps: Sampling rate of the BVP signals in frames per second.
        device: Target device; defaults to CUDA when available, else CPU.

    Returns:
        A dict with aggregate metrics ``mae``, ``rmse``, ``within5`` (percent
        within +/-5 bpm), ``pearson`` (HR correlation), the window count ``n``,
        and the per-window arrays ``hr_pred`` and ``hr_true`` (bpm).
    """
    dev = _resolve_device(device)
    model = model.to(dev)
    was_training = model.training
    model.eval()

    pred_hr: List[float] = []
    true_hr: List[float] = []
    try:
        with torch.no_grad():
            for batch in loader:
                inputs, targets = _unpack_batch(batch)
                outputs = model(inputs.to(dev))
                pred_bvp = outputs.detach().cpu().numpy()
                true_bvp = targets.detach().cpu().numpy()
                batch_pred, batch_true = _window_hrs(pred_bvp, true_bvp, fps)
                pred_hr.extend(batch_pred)
                true_hr.extend(batch_true)
    finally:
        model.train(was_training)

    pred_arr = np.asarray(pred_hr, dtype=np.float64)
    true_arr = np.asarray(true_hr, dtype=np.float64)

    # Drop windows where either HR estimate is non-finite (e.g. flat signal).
    valid = np.isfinite(pred_arr) & np.isfinite(true_arr)
    pred_arr = pred_arr[valid]
    true_arr = true_arr[valid]
    if pred_arr.size == 0:
        return _empty_result()

    return {
        "mae": mae(pred_arr, true_arr),
        "rmse": rmse(pred_arr, true_arr),
        "within5": accuracy_within(pred_arr, true_arr, tol_bpm=_WITHIN_TOL_BPM),
        "pearson": _pearson(pred_arr, true_arr),
        "n": int(pred_arr.size),
        "hr_pred": pred_arr,
        "hr_true": true_arr,
    }
