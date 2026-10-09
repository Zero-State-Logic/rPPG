"""Shape, gradient-flow, and split-integrity tests for the deep stack."""
from __future__ import annotations

import pytest
import torch

from rppg.data.splits import (cross_dataset_split, loso_folds,
                              subject_independent_split)
from rppg.models.losses import FrequencyMSELoss, NegPearsonLoss
from rppg.models.physnet import PhysNet
from rppg.models.proposnet import ProPOSNet
from rppg.models.tscan import TSCAN


@pytest.mark.parametrize("model_cls", [ProPOSNet, PhysNet, TSCAN])
def test_model_forward_shape_and_grad(model_cls):
    model = model_cls()
    x = torch.randn(2, 3, 16, 72, 72)
    out = model(x)
    assert out.shape == (2, 16)
    NegPearsonLoss()(out, torch.randn(2, 16)).backward()
    assert any(p.grad is not None for p in model.parameters())


def test_negpearson_zero_for_identical_signals():
    sig = torch.randn(2, 32)
    assert float(NegPearsonLoss()(sig, sig.clone())) == pytest.approx(0.0, abs=1e-4)


def test_frequency_loss_nonnegative():
    assert float(FrequencyMSELoss()(torch.randn(2, 32), torch.randn(2, 32))) >= 0.0


def test_subject_independent_split_disjoint():
    ids = [f"s{i}" for i in range(20)]
    split = subject_independent_split(ids, seed=0)
    assert not (set(split["train"]) & set(split["test"]))
    assert not (set(split["val"]) & set(split["test"]))
    assert not (set(split["train"]) & set(split["val"]))
    assert len(split["train"]) + len(split["val"]) + len(split["test"]) == 20


def test_loso_and_cross_dataset_splits():
    ids = [f"s{i}" for i in range(5)]
    folds = loso_folds(ids)
    assert len(folds) == 5
    cross = cross_dataset_split(["a1", "a2"], ["b1", "b2"])
    assert cross["train"] == ["a1", "a2"] and cross["test"] == ["b1", "b2"]
