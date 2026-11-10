import keras
import numpy as np
import pytest

from src.architectures import DeepSpectra_architecture, IPA_architecture
from src.architectures.DeepSpectra_architecture import DeepSpectra
from src.architectures.IPA_architecture import IPA

FULL_RANGE, REDUCED_RANGE = 4150, 3890

# Reference parameter counts: a change means the architecture changed.
PARAM_COUNTS = {
    (IPA, REDUCED_RANGE): 200_161,
    (IPA, FULL_RANGE): 211_681,
    (DeepSpectra, REDUCED_RANGE): 36_265,
    (DeepSpectra, FULL_RANGE): 38_697,
}

MODELS = [IPA, DeepSpectra]


def build(model_class, n_features=REDUCED_RANGE, seed=1, **kwargs):
    keras.utils.set_random_seed(seed)
    model = model_class(seed_value=seed, regularization_factor=0.01, **kwargs)
    model(np.zeros((1, n_features, 1), dtype=np.float32))
    return model


def basic_convs(model):
    """All BasicConv1D blocks of a model, whatever their nesting."""
    classes = (IPA_architecture.BasicConv1D, DeepSpectra_architecture.BasicConv1D)
    return [layer for layer in model._flatten_layers() if isinstance(layer, classes)]


@pytest.mark.parametrize("model_class", MODELS)
@pytest.mark.parametrize("n_features", [REDUCED_RANGE, FULL_RANGE])
@pytest.mark.parametrize("batch_size", [1, 5])
def test_output_shape(model_class, n_features, batch_size):
    model = build(model_class, n_features)
    out = model(np.random.rand(batch_size, n_features, 1).astype(np.float32))
    assert tuple(out.shape) == (batch_size, 1)


@pytest.mark.parametrize("model_class, n_features", PARAM_COUNTS.keys())
def test_parameter_count(model_class, n_features):
    assert build(model_class, n_features).count_params() == PARAM_COUNTS[(model_class, n_features)]


@pytest.mark.parametrize("model_class", MODELS)
def test_leaky_relu_slope_matches_paper(model_class):
    # Keras 3 changed the LeakyReLU default from 0.3 to 0.2
    convs = basic_convs(build(model_class))
    assert convs
    assert all(conv.activation.negative_slope == pytest.approx(0.3) for conv in convs)


@pytest.mark.parametrize("model_class", MODELS)
def test_inference_is_deterministic(model_class):
    model = build(model_class)
    x = np.random.rand(4, REDUCED_RANGE, 1).astype(np.float32)
    np.testing.assert_array_equal(model.predict(x, verbose=0), model.predict(x, verbose=0))


@pytest.mark.parametrize("model_class", MODELS)
def test_same_seed_gives_same_weights(model_class):
    for w1, w2 in zip(build(model_class).get_weights(), build(model_class).get_weights()):
        np.testing.assert_array_equal(w1, w2)


@pytest.mark.parametrize("model_class", MODELS)
def test_dropout_rate_is_applied(model_class):
    assert build(model_class, dropout_rate=0.5).dropout.rate == 0.5


def test_ipa_conv_initializers_use_the_seed():
    convs = basic_convs(build(IPA, seed=7))
    assert all(conv.conv.kernel_initializer.seed == 7 for conv in convs)


def test_deepspectra_conv_initializers_use_the_seed():
    convs = basic_convs(build(DeepSpectra, seed=7))
    assert all(conv.conv.kernel_initializer.seed == 7 for conv in convs)
