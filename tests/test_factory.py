import pytest

from src.model_utils.base_model_config import BaseModelConfig
from src.model_utils.base_model_pipeline import BaseModelPipeline
from src.model_utils.model_type import ModelType


@pytest.mark.parametrize("name, model_type", [("IPA", ModelType.IPA), ("DeepSpectra", ModelType.DEEPSPECTRA)])
def test_create_from_string(calibration, name, model_type):
    pipeline = calibration.ModelFactory.create_from_string(name)
    assert isinstance(pipeline, calibration.ModelFactory._registry[model_type])
    assert pipeline.config.model_type is model_type
    assert pipeline.config.model_name == name


def test_create_from_unknown_string(calibration):
    with pytest.raises(ValueError, match="Unknown model name"):
        calibration.ModelFactory.create_from_string("ResNet")


def test_create_pipeline_uses_given_config(calibration):
    config = calibration.IPAConfig(epochs=3)
    assert calibration.ModelFactory.create_pipeline(ModelType.IPA, config).config is config


def test_every_model_type_is_registered(calibration):
    assert set(calibration.ModelFactory.list_available_models()) == {mt.value for mt in ModelType}
    assert set(calibration.ModelFactory._config_registry) == set(ModelType)


def test_register_model(calibration, monkeypatch):
    factory = calibration.ModelFactory
    monkeypatch.setattr(factory, "_registry", dict(factory._registry))
    monkeypatch.setattr(factory, "_config_registry", dict(factory._config_registry))

    class DummyPipeline(BaseModelPipeline):
        def build_model(self):
            return None

    factory.register_model(ModelType.IPA, DummyPipeline, BaseModelConfig)
    assert isinstance(factory.create_pipeline(ModelType.IPA), DummyPipeline)


def test_default_configs(calibration):
    ipa, ds = calibration.IPAConfig(), calibration.DeepSpectraConfig()
    assert (ipa.loss_function, ipa.learning_rate, ipa.batch_size) == ("mae", 0.001, 16)
    assert (ds.loss_function, ds.learning_rate, ds.batch_size) == ("mse", 0.01, 32)


def test_model_type_cannot_be_overridden(calibration):
    with pytest.raises(TypeError):
        calibration.IPAConfig(model_type=ModelType.DEEPSPECTRA)


def test_unknown_config_key_is_rejected(calibration):
    with pytest.raises(TypeError):
        calibration.IPAConfig(**{"learning_rte": 0.1})


@pytest.mark.parametrize("model_type", list(ModelType))
def test_build_model_is_compiled_with_config(make_pipeline, model_type):
    pipeline = make_pipeline(model_type)
    model = pipeline.build_model()
    assert model.loss == pipeline.config.loss_function
    assert model.optimizer.learning_rate.numpy() == pytest.approx(pipeline.config.learning_rate)
