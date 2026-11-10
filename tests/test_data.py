"""Integrity checks on the dataset shipped in data/."""
from pathlib import Path

import numpy as np
import pytest
from scipy.io import loadmat

DATA = Path(__file__).resolve().parents[1] / "data"
N_TRAIN, N_TEST = 174, 75
N_FEATURES = {"full_range": 4150, "reduced_range": 3890}


@pytest.fixture(scope="module", params=N_FEATURES, ids=list(N_FEATURES))
def mat(request):
    return request.param, loadmat(DATA / f"raw-kennard-{request.param}.mat")


@pytest.fixture(scope="module")
def y():
    return {split: np.genfromtxt(DATA / f"y_{split}_scaled.csv", delimiter=",") for split in ("train", "test")}


def test_mat_content(mat):
    name, d = mat
    assert d["X_train"].shape == (N_TRAIN, N_FEATURES[name])
    assert d["X_test"].shape == (N_TEST, N_FEATURES[name])
    assert d["W"].shape == (1, N_FEATURES[name])
    assert np.isfinite(d["X_train"]).all() and np.isfinite(d["X_test"]).all()


def test_wavelengths_are_sorted(mat):
    w = mat[1]["W"].ravel()
    assert np.all(np.diff(w) > 0) or np.all(np.diff(w) < 0)


def test_reduced_range_is_a_subset_of_full_range():
    full = loadmat(DATA / "raw-kennard-full_range.mat")
    reduced = loadmat(DATA / "raw-kennard-reduced_range.mat")
    assert np.isin(reduced["W"], full["W"]).all()


def test_y_csv(y):
    assert y["train"].shape == (N_TRAIN,)
    assert y["test"].shape == (N_TEST,)
    assert np.isfinite(y["train"]).all() and np.isfinite(y["test"]).all()


def test_y_train_is_standardised(y):
    assert y["train"].mean() == pytest.approx(0, abs=1e-6)
    assert y["train"].std() == pytest.approx(1, abs=1e-6)


def test_csv_y_and_mat_y_are_the_same_samples(mat, y):
    """CSV and .mat targets are scaled differently but must be the same samples, in the same order."""
    _, d = mat
    for split in ("train", "test"):
        assert np.corrcoef(y[split], d[f"y_{split}"].ravel())[0, 1] > 0.9999
