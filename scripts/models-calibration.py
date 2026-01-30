#!/usr/bin/env python
# coding: utf-8
import logging
from typing import Optional, Dict, Type
from dataclasses import dataclass, field
import warnings

import tensorflow as tf

from src.architectures.DeepSpectra_architecture import DeepSpectra
from src.architectures.IPA_architecture import IPA
from src.model_utils.base_model_config import BaseModelConfig
from src.model_utils.base_model_pipeline import BaseModelPipeline
from src.model_utils.model_type import ModelType

warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class IPAConfig(BaseModelConfig):
    model_type: ModelType = field(default=ModelType.IPA, init=False)
    model_name: str = "IPA"
    learning_rate: float = 0.001
    batch_size: int = 16
    regularization_coef: float = 0.0095
    dropout_rate: float = 0.2
    loss_function: str = 'mae'

@dataclass
class DeepSpectraConfig(BaseModelConfig):
    model_type: ModelType = field(default=ModelType.DEEPSPECTRA, init=False)
    model_name: str = "DeepSpectra"
    learning_rate: float = 0.01
    batch_size: int = 32
    regularization_coef: float = 0.001
    dropout_rate: float = 0.2
    loss_function: str = 'mse'


class IPAModelPipeline(BaseModelPipeline):
    """Pipeline for the IPA model"""

    def __init__(self, config: IPAConfig):
        super().__init__(config)
        self.config: IPAConfig = config

    def build_model(self):
        """Build and compile the IPA model."""
        logger.info("Building IPA model...")

        model = IPA(
            seed_value=self.config.seed_value,
            regularization_factor=self.config.regularization_coef
        )

        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=self.config.learning_rate,
            decay_steps=self.config.decay_steps,
            decay_rate=self.config.decay_rate
        )

        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        model.compile(optimizer=optimizer, loss=self.config.loss_function, metrics=['mse'])

        logger.info("IPA model built and compiled")
        return model


class DeepSpectraModelPipeline(BaseModelPipeline):
    """Pipeline for the DeepSpectra model"""

    def __init__(self, config: DeepSpectraConfig):
        super().__init__(config)
        self.config: DeepSpectraConfig = config

    def build_model(self):
        """Build and compile the DeepSpectra model."""
        logger.info("Building DeepSpectra model...")

        model = DeepSpectra(
            seed_value=self.config.seed_value,
            regularization_factor=self.config.regularization_coef
        )

        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=self.config.learning_rate,
            decay_steps=self.config.decay_steps,
            decay_rate=self.config.decay_rate
        )

        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        model.compile(optimizer=optimizer, loss=self.config.loss_function, metrics=['mse'])

        logger.info("DeepSpectra model built and compiled")
        return model


class ModelFactory:
    """Factory for creating model pipelines."""

    # Mapping models to their respective pipelines
    _registry: Dict[ModelType, Type[BaseModelPipeline]] = {
        ModelType.IPA: IPAModelPipeline, 
        ModelType.DEEPSPECTRA: DeepSpectraModelPipeline, 
    }

    # Mapping models to their respective config
    _config_registry: Dict[ModelType, Type[BaseModelConfig]] = {
        ModelType.IPA: IPAConfig,
        ModelType.DEEPSPECTRA: DeepSpectraConfig
    }

    @classmethod
    def create_pipeline(
        cls,
        model_type: ModelType,
        config: Optional[BaseModelConfig] = None
    ) -> BaseModelPipeline:
        """Create a model pipeline instance
        
        Args:
            model_type: Type of model to create
            config: Optional config. If None, uses default for model type
        
        Returns:
            Instantiated model pipeline
        
        Raises:
            ValueError: If model type is not registered
        """
        if model_type not in cls._registry:
            available = ', '.join([mt.value for mt in cls._registry.keys()])
            raise ValueError(
                f"Unknown model type: {model_type}. "
                f"Available types: {available}"
            )

        if config is None:
            config_class = cls._config_registry[model_type]
            config = config_class()
        
        pipeline_class = cls._registry[model_type]
        logger.info(f"Creating {model_type.value} pipeline")
        
        return pipeline_class(config)

    @classmethod
    def create_from_string(
        cls, 
        model_name: str, 
        config: Optional[BaseModelConfig] = None
    ) -> BaseModelPipeline:
        """Create a model pipeline from a string name.
        
        Args:
            model_name: Name of the model (e.g., "IPA", "DeepSpectra")
            config: Optional configuration
            
        Returns:
            Instantiated model pipeline
        """
        try:
            model_type = ModelType(model_name)
            return cls.create_pipeline(model_type, config)
        except ValueError:
            available = ', '.join([mt.value for mt in ModelType])
            raise ValueError(
                f"Unknown model name: {model_name}. "
                f"Available models: {available}"
            )
    
    @classmethod
    def register_model(
        cls, 
        model_type: ModelType, 
        pipeline_class: Type[BaseModelPipeline],
        config_class: Type[BaseModelConfig]
    ) -> None:
        """Register a new model type.
        
        Args:
            model_type: Model type enum
            pipeline_class: Pipeline class for this model
            config_class: Config class for this model
        """
        cls._registry[model_type] = pipeline_class
        cls._config_registry[model_type] = config_class
        logger.info(f"Registered new model type: {model_type.value}")
    
    @classmethod
    def list_available_models(cls) -> list[str]:
        """Get list of available model types."""
        return [model_type.value for model_type in cls._registry.keys()]


def basic_usage():
    """Basic usage with default configuration"""

    # Launch IPA and collect results
    ipa_pipeline = ModelFactory.create_from_string("IPA")
    ipa_results = ipa_pipeline.run_pipeline()
    print(f"Model name: {ModelType.IPA.name}, RMSEP: {ipa_results['RMSEP']:.4f}, R2: {ipa_results['R2P']:.4f}")

    # # Launch DeepSpectra and collect results
    ds_pipeline = ModelFactory.create_pipeline(ModelType.DEEPSPECTRA)
    ds_results = ds_pipeline.run_pipeline()
    print(f"Model name: {ModelType.DEEPSPECTRA.name}, RMSEP: {ds_results['RMSEP']:.4f}, R2: {ds_results['R2P']:.4f}")

def compare_models():
    """Compare multiple models."""
    
    results = {}
    for model_type in ModelType:
        pipeline = ModelFactory.create_pipeline(model_type)
        model_results = pipeline.run_pipeline()
        results[model_type.value] = {
            'RMSEP': model_results['RMSEP'],
            'R2P': model_results['R2P']
        }
    
    # Print comparison
    print("\nModel Comparison:")
    print("-" * 40)
    for model_name, metrics in results.items():
        print(f"{model_name:15s} - RMSEP: {metrics['RMSEP']:.4f}, R²: {metrics['R2P']:.4f}")

def config_from_dict():
    """Create config from dictionary (possible to expand to JSON/YAML configs)."""
    
    config_dict = {
        # 'data_path': 'data/raw-kennard-full_range.mat', # Another choice for model training/evaluation.
        'data_path': 'data/raw-kennard-reduced_range.mat',
        'learning_rate': 0.001,
        'batch_size': 16,
        'epochs': 200,
        'regularization_coef': 0.01
    }
    
    config = IPAConfig(**config_dict)
    pipeline = ModelFactory.create_pipeline(ModelType.IPA, config)
    results = pipeline.run_pipeline(force_retrain=False)
    print(f"Model name: {ModelType.IPA.name}, RMSEP: {results['RMSEP']:.4f}, R2: {results['R2P']:.4f}")


def main():
    """
    Select which function suits your needs
    """
    # basic_usage()
    compare_models()
    # config_from_dict()


if __name__ == "__main__":
    main()
