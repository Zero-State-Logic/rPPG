"""Dataset splitting for subject-independent and cross-dataset rPPG evaluation.

All splits operate on plain lists of subject-id strings and are fully
deterministic: randomness is confined to a local :class:`random.Random`
instance seeded per call, so results never depend on the global RNG, wall-clock
time, or NumPy state. No file writes or network access occur here.

Three regimes are supported:

* **Subject-independent** -- disjoint train/val/test subject lists from a single
  pool (no subject appears in more than one split).
* **Leave-one-subject-out (LOSO)** -- one fold per subject, that subject held out
  for test and every other subject used for training.
* **Cross-dataset ("uneven data")** -- train on one dataset's subjects, test on
  another dataset's subjects.
"""
from __future__ import annotations

import random
from typing import List, Tuple

__all__ = [
    "subject_independent_split",
    "loso_folds",
    "cross_dataset_split",
]


def _dedupe_preserve_order(subject_ids: List[str]) -> List[str]:
    """Return subject ids with duplicates removed, keeping first-seen order.

    Args:
        subject_ids: Subject-id strings, possibly with repeats.

    Returns:
        A new list containing each id exactly once in order of first appearance.
    """
    seen: set[str] = set()
    unique: List[str] = []
    for sid in subject_ids:
        if sid not in seen:
            seen.add(sid)
            unique.append(sid)
    return unique


def _validate_fracs(val_frac: float, test_frac: float) -> None:
    """Validate that split fractions are usable.

    Args:
        val_frac: Fraction of subjects for the validation split.
        test_frac: Fraction of subjects for the test split.

    Raises:
        ValueError: If either fraction is negative or their sum leaves no room
            for a non-empty training split.
    """
    if val_frac < 0.0 or test_frac < 0.0:
        raise ValueError("val_frac and test_frac must be non-negative")
    if val_frac + test_frac >= 1.0:
        raise ValueError("val_frac + test_frac must be < 1.0 to leave a train split")


def subject_independent_split(
    subject_ids: List[str],
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
) -> dict[str, List[str]]:
    """Partition subjects into disjoint train/val/test id lists.

    The input is de-duplicated, shuffled with a locally seeded RNG, then sliced
    so that no subject appears in more than one split. Split sizes use floor
    rounding for val/test; any remainder is assigned to train, which therefore
    always receives at least the leftover subjects.

    Args:
        subject_ids: Pool of subject-id strings (duplicates are collapsed).
        val_frac: Fraction of unique subjects placed in the validation split.
        test_frac: Fraction of unique subjects placed in the test split.
        seed: Seed for the local :class:`random.Random` used to shuffle.

    Returns:
        A dict with keys ``"train"``, ``"val"`` and ``"test"`` mapping to
        pairwise-disjoint lists of subject ids whose union is the unique input.

    Raises:
        ValueError: If the fractions are invalid (see :func:`_validate_fracs`).
    """
    _validate_fracs(val_frac, test_frac)

    unique = _dedupe_preserve_order(list(subject_ids))
    rng = random.Random(seed)
    shuffled = unique[:]
    rng.shuffle(shuffled)

    n_total = len(shuffled)
    n_test = int(n_total * test_frac)
    n_val = int(n_total * val_frac)
    if n_total >= 3:  # guarantee non-empty held-out splits for small pools
        n_test = max(1, n_test)
        n_val = max(1, n_val)
        if n_test + n_val > n_total - 1:  # but always keep >=1 for train
            n_val = n_total - 1 - n_test

    test_ids = shuffled[:n_test]
    val_ids = shuffled[n_test:n_test + n_val]
    train_ids = shuffled[n_test + n_val:]

    return {"train": train_ids, "val": val_ids, "test": test_ids}


def loso_folds(subject_ids: List[str]) -> List[Tuple[List[str], List[str]]]:
    """Build leave-one-subject-out folds.

    For each unique subject (in first-seen order) one fold is produced with that
    subject held out for testing and all other subjects used for training. No
    randomness is involved, so the fold order is fully deterministic.

    Args:
        subject_ids: Pool of subject-id strings (duplicates are collapsed).

    Returns:
        A list of ``(train_ids, test_ids)`` tuples, one per unique subject. Each
        ``test_ids`` is a single-element list ``[subject]`` and ``train_ids`` is
        every other unique subject in first-seen order.
    """
    unique = _dedupe_preserve_order(list(subject_ids))
    folds: List[Tuple[List[str], List[str]]] = []
    for held_out in unique:
        train_ids = [sid for sid in unique if sid != held_out]
        folds.append((train_ids, [held_out]))
    return folds


def cross_dataset_split(
    ds_a_ids: List[str],
    ds_b_ids: List[str],
) -> dict[str, List[str]]:
    """Split across datasets: train on dataset A, test on dataset B.

    This models the "uneven data" protocol where training and evaluation draw
    from different corpora. Each dataset's ids are de-duplicated independently;
    no attempt is made to exclude overlap between the two datasets, since they
    are treated as distinct sources.

    Args:
        ds_a_ids: Subject-id strings from the training dataset (dataset A).
        ds_b_ids: Subject-id strings from the test dataset (dataset B).

    Returns:
        A dict with ``"train"`` set to dataset A's unique ids and ``"test"`` set
        to dataset B's unique ids.
    """
    return {
        "train": _dedupe_preserve_order(list(ds_a_ids)),
        "test": _dedupe_preserve_order(list(ds_b_ids)),
    }
