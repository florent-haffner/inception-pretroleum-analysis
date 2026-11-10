import importlib
import os

# Must be set before TensorFlow / matplotlib are imported
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import pytest
from scipy.io import savemat

N_FEATURES = 300  # Small spectra keep the tests fast
N_TRAIN, N_TEST = 24, 10


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: training on the real dataset (run with `make test-all`)")
    # Harmless noise: plt.show() without a display, Keras/numpy 2 internals, optimizer state not reloaded
    for message in (
        "FigureCanvasAgg is non-interactive",
        "__array__ implementation doesn't accept a copy keyword",
        "Skipping variable loading for optimizer",
    ):
        config.addinivalue_line("filterwarnings", f"ignore:{message}")


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    import matplotlib.pyplot as plt
    plt.close("all")


@pytest.fixture(scope="session")
def calibration():
    """The calibration script (its file name contains a dash, so it is imported this way)."""
    return importlib.import_module("scripts.models-calibration")


@pytest.fixture
def synthetic_paths(tmp_path):
    """Write a small dataset with the same layout as the real one and return the config paths."""
    rng = np.random.default_rng(0)
    X_train = rng.normal(5.0, 2.0, size=(N_TRAIN, N_FEATURES))
    X_test = rng.normal(5.0, 2.0, size=(N_TEST, N_FEATURES))
    y_train = X_train[:, :10].mean(axis=1)
    y_test = X_test[:, :10].mean(axis=1)

    data_path = tmp_path / "data.mat"
    savemat(data_path, {"X_train": X_train, "X_test": X_test})
    y_train_path = tmp_path / "y_train.csv"
    y_test_path = tmp_path / "y_test.csv"
    np.savetxt(y_train_path, y_train, delimiter=",")
    np.savetxt(y_test_path, y_test, delimiter=",")

    return {
        "data_path": str(data_path),
        "y_train_path": str(y_train_path),
        "y_test_path": str(y_test_path),
        "model_path": str(tmp_path / "model"),
    }


@pytest.fixture
def make_pipeline(calibration, synthetic_paths):
    """Build a pipeline for a model type on the synthetic dataset, with a short training."""
    def _make(model_type, **overrides):
        config_class = calibration.ModelFactory._config_registry[model_type]
        params = {**synthetic_paths, "epochs": 2, "batch_size": 8, **overrides}
        return calibration.ModelFactory.create_pipeline(model_type, config_class(**params))
    return _make
