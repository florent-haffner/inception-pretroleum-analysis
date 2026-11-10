import numpy as np
import pytest
from scipy.io import savemat

from src.model_utils.model_type import ModelType

from conftest import N_FEATURES, N_TEST, N_TRAIN


@pytest.fixture
def ipa(make_pipeline):
    return make_pipeline(ModelType.IPA)


# --- load_data -------------------------------------------------------------

def test_load_data_shapes(ipa):
    X_train, y_train, X_test, y_test = ipa.load_data()
    assert X_train.shape == (N_TRAIN, N_FEATURES)
    assert X_test.shape == (N_TEST, N_FEATURES)
    assert y_train.shape == (N_TRAIN,)
    assert y_test.shape == (N_TEST,)


@pytest.mark.parametrize("missing", ["data_path", "y_train_path", "y_test_path"])
def test_load_data_missing_file(make_pipeline, missing, tmp_path):
    pipeline = make_pipeline(ModelType.IPA, **{missing: str(tmp_path / "missing")})
    with pytest.raises(FileNotFoundError):
        pipeline.load_data()


def test_load_data_size_mismatch(ipa, synthetic_paths):
    savemat(synthetic_paths["data_path"], {
        "X_train": np.zeros((N_TRAIN + 1, N_FEATURES)),
        "X_test": np.zeros((N_TEST, N_FEATURES)),
    })
    with pytest.raises(ValueError, match="size mismatch"):
        ipa.load_data()


# --- preprocessing ---------------------------------------------------------

def test_preprocess_x_does_not_modify_inputs(ipa):
    X_train, _, X_test, _ = ipa.load_data()
    X_train_copy, X_test_copy = X_train.copy(), X_test.copy()
    ipa.preprocess_X(X_train, X_test)
    np.testing.assert_array_equal(X_train, X_train_copy)
    np.testing.assert_array_equal(X_test, X_test_copy)


def test_preprocess_x_adds_channel_axis(ipa):
    X_train, _, X_test, _ = ipa.load_data()
    X_train_p, X_test_p = ipa.preprocess_X(X_train, X_test)
    assert X_train_p.shape == (N_TRAIN, N_FEATURES, 1)
    assert X_test_p.shape == (N_TEST, N_FEATURES, 1)


def test_scaler_is_fitted_on_train_only(ipa):
    X_train, _, X_test, _ = ipa.load_data()
    X_train_p, X_test_p = ipa.preprocess_X(X_train, X_test)

    np.testing.assert_allclose(ipa.scalerX.mean_, X_train.mean(axis=0))
    np.testing.assert_allclose(X_train_p.mean(axis=0), 0, atol=1e-10)
    # Test set is transformed with the train statistics, so it is not exactly centred
    expected_test = (X_test - X_train.mean(axis=0)) / X_train.std(axis=0)
    np.testing.assert_allclose(X_test_p[..., 0], expected_test)


def test_preprocess_y_adds_column_axis(ipa):
    _, y_train, _, y_test = ipa.load_data()
    y_train_p, y_test_p = ipa.preprocess_y(y_train, y_test)
    assert y_train_p.shape == (N_TRAIN, 1)
    assert y_test_p.shape == (N_TEST, 1)
    np.testing.assert_array_equal(y_train_p[:, 0], y_train)


# --- model lifecycle ---------------------------------------------------------

def test_predict_without_model_raises(ipa):
    with pytest.raises(ValueError, match="No model"):
        ipa.predict(*[np.zeros(1)] * 4)


def test_save_without_model_raises(ipa):
    with pytest.raises(ValueError, match="No model"):
        ipa.save_model()


def test_load_model_returns_false_when_absent(ipa):
    assert ipa.load_model((N_FEATURES, 1)) is False
    assert ipa.model is None


@pytest.mark.parametrize("model_type", list(ModelType))
def test_save_and_reload_gives_same_predictions(make_pipeline, model_type):
    first = make_pipeline(model_type).run_pipeline()

    reloaded = make_pipeline(model_type)
    second = reloaded.run_pipeline()

    assert "history" in first           # trained
    assert "history" not in second      # reused the saved weights
    np.testing.assert_allclose(first["predictions"], second["predictions"], atol=1e-5)


def test_reported_param_count_excludes_optimizer(make_pipeline, caplog):
    import logging
    pipeline = make_pipeline(ModelType.IPA)
    with caplog.at_level(logging.INFO):
        results = pipeline.run_pipeline()
    weights = sum(int(np.prod(w.shape)) for w in pipeline.model.weights)
    assert results["n_params"] == weights
    assert f"Model parameters: {weights:,}" in caplog.text


def test_force_retrain(make_pipeline):
    make_pipeline(ModelType.IPA).run_pipeline()
    results = make_pipeline(ModelType.IPA).run_pipeline(force_retrain=True)
    assert "history" in results


def test_weights_file_location(make_pipeline, synthetic_paths):
    from pathlib import Path
    make_pipeline(ModelType.DEEPSPECTRA).run_pipeline()
    assert (Path(synthetic_paths["model_path"]) / "DeepSpectra.weights.h5").exists()


def test_training_loss_is_finite(make_pipeline):
    results = make_pipeline(ModelType.IPA, epochs=3).run_pipeline()
    assert np.isfinite(results["history"]["loss"]).all()
    assert np.isfinite(results["RMSEP"])


def test_same_seed_gives_same_training(make_pipeline, synthetic_paths, tmp_path):
    a = make_pipeline(ModelType.IPA, model_path=str(tmp_path / "a")).run_pipeline()
    b = make_pipeline(ModelType.IPA, model_path=str(tmp_path / "b")).run_pipeline()
    np.testing.assert_allclose(a["history"]["loss"], b["history"]["loss"], rtol=1e-5)


@pytest.mark.parametrize("model_type", list(ModelType))
def test_config_dropout_rate_reaches_model(make_pipeline, model_type):
    model = make_pipeline(model_type, dropout_rate=0.5).build_model()
    assert model.dropout.rate == 0.5


# --- real dataset (slow) -----------------------------------------------------

@pytest.mark.slow
@pytest.mark.parametrize("model_type", list(ModelType))
def test_short_training_on_real_data(calibration, model_type, tmp_path):
    config_class = calibration.ModelFactory._config_registry[model_type]
    pipeline = calibration.ModelFactory.create_pipeline(
        model_type, config_class(epochs=5, model_path=str(tmp_path))
    )
    results = pipeline.run_pipeline()
    loss = results["history"]["loss"]
    assert np.isfinite(loss).all()
    assert loss[-1] < loss[0]
