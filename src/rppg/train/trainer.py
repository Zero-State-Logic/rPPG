"""Device-agnostic training loop for rPPG BVP models.

Trains any model conforming to the uniform deep interface: ``forward(x)``
takes ``x`` of shape ``(B, 3, T, H, W)`` and returns a BVP tensor of shape
``(B, T)``. The loop is framework-plain (Adam + early stopping on validation
loss) and runs unchanged on CPU or CUDA.

No file writes or network access happen at import time.
"""
from __future__ import annotations

import os
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import torch
from torch import nn

BestLoss = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]

_BEST_FILENAME = "best.pt"


def _resolve_device(device: Optional[Any]) -> torch.device:
    """Return an explicit device, defaulting to CUDA when available."""
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _default_loss() -> nn.Module:
    """Lazily build the default :class:`NegPearsonLoss`.

    Imported at call time so this module imports cleanly even before the
    losses module is present.
    """
    from rppg.models.losses import NegPearsonLoss

    return NegPearsonLoss()


def _unpack_batch(
    batch: Any, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract ``(inputs, targets)`` from a batch and move them to ``device``.

    Supports the common collate shapes: a ``(inputs, targets, ...)`` tuple or
    list, or a mapping exposing input/target tensors under conventional keys.
    """
    if isinstance(batch, dict):
        in_key = next(k for k in ("input", "inputs", "x", "frames") if k in batch)
        tgt_key = next(k for k in ("target", "targets", "y", "bvp", "label") if k in batch)
        inputs, targets = batch[in_key], batch[tgt_key]
    elif isinstance(batch, (tuple, list)):
        if len(batch) < 2:
            raise ValueError("Batch must provide at least (inputs, targets).")
        inputs, targets = batch[0], batch[1]
    else:
        raise TypeError(f"Unsupported batch type: {type(batch)!r}")

    inputs = torch.as_tensor(inputs, dtype=torch.float32).to(device)
    targets = torch.as_tensor(targets, dtype=torch.float32).to(device)
    return inputs, targets


def _run_epoch(
    model: nn.Module,
    loader: Any,
    loss_fn: BestLoss,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer],
) -> float:
    """Run a single epoch; train when ``optimizer`` is given, else evaluate.

    Returns the sample-weighted mean loss, or ``nan`` for an empty loader.
    """
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_items = 0
    grad_ctx = torch.enable_grad() if is_train else torch.no_grad()
    with grad_ctx:
        for batch in loader:
            inputs, targets = _unpack_batch(batch, device)
            batch_size = int(inputs.shape[0])
            if is_train:
                optimizer.zero_grad(set_to_none=True)
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            if is_train:
                loss.backward()
                optimizer.step()
            total_loss += float(loss.detach().item()) * batch_size
            total_items += batch_size

    if total_items == 0:
        return float("nan")
    return total_loss / total_items


def _has_batches(loader: Optional[Any]) -> bool:
    """True when ``loader`` is non-None and yields at least one batch."""
    if loader is None:
        return False
    try:
        return len(loader) > 0  # type: ignore[arg-type]
    except TypeError:
        return any(True for _ in loader)


def train_model(
    model: nn.Module,
    train_loader: Any,
    val_loader: Optional[Any] = None,
    epochs: int = 30,
    lr: float = 1e-3,
    device: Optional[Any] = None,
    out_dir: Optional[str] = None,
    loss_fn: Optional[BestLoss] = None,
    patience: int = 5,
) -> Dict[str, Sequence[float]]:
    """Train ``model`` and return the loss history.

    Args:
        model: Module with the uniform ``(B, 3, T, H, W) -> (B, T)`` interface.
        train_loader: Iterable of batches yielding ``(inputs, targets)``.
        val_loader: Optional validation loader used for early stopping.
        epochs: Maximum number of epochs.
        lr: Adam learning rate.
        device: Target device; defaults to CUDA when available, else CPU.
        out_dir: When given, the best state dict is saved to ``out_dir/best.pt``.
        loss_fn: Loss callable; defaults to ``NegPearsonLoss``.
        patience: Early-stop patience (epochs without validation improvement).

    Returns:
        ``{"train_loss": [...], "val_loss": [...]}``.

    Raises:
        ValueError: When ``train_loader`` yields no batches.
    """
    if not _has_batches(train_loader):
        raise ValueError("train_loader is empty; nothing to train on.")

    device = _resolve_device(device)
    model = model.to(device)
    if loss_fn is None:
        loss_fn = _default_loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    use_val = _has_batches(val_loader)
    history: Dict[str, list] = {"train_loss": [], "val_loss": []}
    best_metric = float("inf")
    best_state: Optional[Dict[str, torch.Tensor]] = None
    epochs_without_improve = 0

    for epoch in range(1, epochs + 1):
        train_loss = _run_epoch(model, train_loader, loss_fn, device, optimizer)
        val_loss = (
            _run_epoch(model, val_loader, loss_fn, device, optimizer=None)
            if use_val
            else float("nan")
        )
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        print(
            f"Epoch {epoch:3d}/{epochs} | "
            f"train_loss={train_loss:.6f} | val_loss={val_loss:.6f}"
        )

        # Early stopping tracks val loss when available, else train loss.
        monitored = val_loss if use_val else train_loss
        if monitored < best_metric:
            best_metric = monitored
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
            epochs_without_improve = 0
        else:
            epochs_without_improve += 1
            if epochs_without_improve >= patience:
                print(f"Early stopping at epoch {epoch} (patience={patience}).")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
        if out_dir is not None:
            os.makedirs(out_dir, exist_ok=True)
            torch.save(best_state, os.path.join(out_dir, _BEST_FILENAME))

    return history


def cpu_smoke(model: nn.Module) -> bool:
    """Run one forward+backward pass on CPU and verify the output shape.

    Uses random input ``(2, 3, 16, 72, 72)`` and random label ``(2, 16)`` with a
    simple differentiable loss, so it exercises the model without depending on
    the losses module. Returns ``True`` on success; asserts otherwise.
    """
    device = torch.device("cpu")
    model = model.to(device)
    model.train()

    inputs = torch.randn(2, 3, 16, 72, 72, device=device)
    targets = torch.randn(2, 16, device=device)

    outputs = model(inputs)
    assert tuple(outputs.shape) == (2, 16), (
        f"Expected output shape (2, 16), got {tuple(outputs.shape)}."
    )

    loss = ((outputs - targets) ** 2).mean()
    model.zero_grad(set_to_none=True)
    loss.backward()
    return True
