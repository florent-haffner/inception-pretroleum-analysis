import numpy as np
import pytest
from sklearn.metrics import r2_score

from src.utils.plot_utils import plot_err, plot_history, plot_val
from src.utils.regression_utils import cnn_prediction


class FakeModel:
    """Returns fixed predictions and a fixed training MSE."""

    def __init__(self, predictions, train_mse):
        self.predictions = predictions
        self.train_mse = train_mse

    def predict(self, X):
        return self.predictions

    def evaluate(self, X, y, verbose=0):
        return [0.0, self.train_mse]


def test_cnn_prediction_metrics():
    y_test = np.array([[1.0], [2.0], [3.0], [4.0]])
    y_pred = y_test + np.array([[0.5], [-0.5], [0.5], [-0.5]])

    rmsep, r2, preds = cnn_prediction(FakeModel(y_pred, train_mse=0.04), None, None, None, y_test)

    assert rmsep == pytest.approx(0.5)
    assert r2 == pytest.approx(r2_score(y_test, y_pred))
    np.testing.assert_array_equal(preds, y_pred)


def test_cnn_prediction_perfect_model():
    y_test = np.linspace(-1, 1, 10)[:, None]
    rmsep, r2, _ = cnn_prediction(FakeModel(y_test, train_mse=0.01), None, None, None, y_test)
    assert rmsep == 0
    assert r2 == pytest.approx(1.0)


@pytest.mark.filterwarnings("error", "ignore:FigureCanvasAgg is non-interactive")
@pytest.mark.parametrize("offsets", [[0.1, 0.2, 5.0], [5.0, 0.1, 0.2], [0.1, 0.2, 0.3]])
def test_plot_val_colours_points_by_interval(offsets):
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt

    labels = np.array([[10.0], [11.0], [12.0]])
    preds = labels + np.array(offsets)[:, None]
    plot_val(labels, preds, rmse_limit=2.8)

    colours = [mcolors.to_hex(c) for c in plt.gca().collections[0].get_facecolors()]
    expected = ["#c05e31" if abs(o) >= 2.8 else "#212c3d" for o in offsets]
    assert colours == expected


def test_plots_run_without_display():
    rng = np.random.default_rng(0)
    labels = rng.uniform(8, 15, size=(30, 1))
    preds = labels + rng.normal(0, 1, size=(30, 1))

    class History:
        history = {"loss": [3, 2, 1], "val_loss": [3.5, 2.5, 1.5]}

    plot_val(labels, preds)
    plot_err(labels, preds)
    plot_history(History())
