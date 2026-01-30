#!/usr/bin/env python
# coding: utf-8
from abc import ABC, abstractmethod
from enum import Enum
import os
import logging
from copy import deepcopy
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, Type
from dataclasses import dataclass, field
import warnings

import numpy as np
from scipy.io import loadmat
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tqdm.keras import TqdmCallback

from src.architectures.DeepSpectra_architecture import DeepSpectra
from src.architectures.IPA_architecture import IPA
from src.utils.plot_utils import plot_history
from src.utils.regression_utils import cnn_prediction

warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ModelType(Enum):
    """ Enumerate all available models """
    IPA = "IPA"
    DEEPSPECTRA = "DeepSpectra"

@dataclass
class BaseModelConfig:
    """Configuration for model training and inference."""
    # Paths
    data_path: str = "data/raw-kennard-reduced_range.mat"
    # Alternative: Use full range dataset
    # data_path = "data/raw-kennard-full_range.mat"

    y_train_path: str = "data/y_train_scaled.csv"
    y_test_path: str = "data/y_test_scaled.csv"
    model_path: str = "model/"
    
    # Training hyperparameters
    learning_rate: float = 0.001
    batch_size: int = 16
    epochs: int = 500
    
    # Regularization
    regularization_coef: float = 0.0095
    dropout_rate: float = 0.2
    
    # Other settings
    seed_value: int = 1
    early_stopping_patience: int = 50
    decay_steps: int = 10000
    decay_rate: float = 0.001

    # Model type and name
    model_name: str = "model"
    model_type: ModelType = ModelType.IPA


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


class BaseModelPipeline(ABC):
    """Abstract class for pipeline for all model training and inference."""
    
    def __init__(self, config: BaseModelConfig):
        """Initialize the pipeline with configuration.
        
        Args:
            config: ModelConfig instance with all parameters
        """
        self.config = config
        self.model = None
        self.scalerX = StandardScaler()
        self._set_seeds()
        self._configure_gpu()
        
    def _set_seeds(self) -> None:
        """Set random seeds for reproducibility."""
        np.random.seed(self.config.seed_value)
        tf.random.set_seed(self.config.seed_value)
        tf.keras.utils.set_random_seed(self.config.seed_value)
        logger.info(f"Random seeds set to {self.config.seed_value}")
        
    def _configure_gpu(self) -> None:
        """Configure GPU memory growth if available."""
        try:
            physical_devices = tf.config.experimental.list_physical_devices('GPU')
            if physical_devices:
                tf.config.experimental.set_memory_growth(physical_devices[0], True)
                logger.info(f"GPU memory growth activated for {len(physical_devices)} device(s)")
            else:
                logger.info("No GPU devices found, running on CPU")
        except Exception as e:
            logger.warning(f"Error configuring GPU: {e}")
    
    def load_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Load and validate training and test data.
        
        Returns:
            Tuple of (X_train, y_train, X_test, y_test)
            
        Raises:
            FileNotFoundError: If data files don't exist
            ValueError: If data shapes are incompatible
        """
        logger.info("Loading data...")
        
        # Validate file existence
        if not os.path.exists(self.config.data_path):
            raise FileNotFoundError(f"Data file not found: {self.config.data_path}")
        if not os.path.exists(self.config.y_train_path):
            raise FileNotFoundError(f"Y train file not found: {self.config.y_train_path}")
        if not os.path.exists(self.config.y_test_path):
            raise FileNotFoundError(f"Y test file not found: {self.config.y_test_path}")
        
        # Load data
        try:
            X_data = loadmat(self.config.data_path)
            X_train = X_data['X_train']
            X_test = X_data['X_test']
            
            y_train = np.genfromtxt(self.config.y_train_path, delimiter=',')
            y_test = np.genfromtxt(self.config.y_test_path, delimiter=',')
            
            # Validate shapes
            if X_train.shape[0] != y_train.shape[0]:
                raise ValueError(f"X_train and y_train size mismatch: {X_train.shape[0]} vs {y_train.shape[0]}")
            if X_test.shape[0] != y_test.shape[0]:
                raise ValueError(f"X_test and y_test size mismatch: {X_test.shape[0]} vs {y_test.shape[0]}")
            
            logger.info(f"Data loaded - Train: {X_train.shape}, Test: {X_test.shape}")
            return X_train, y_train, X_test, y_test
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def preprocess_X(
        self, 
        X_train: np.ndarray, 
        X_test: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Standardize and reshape input data in an immutable fashion.
        
        Args:
            X_train: Training features
            X_test: Test features
            
        Returns:
            Tuple of (X_train_processed, X_test_processed)
        """
        logger.info("Preprocessing X...")
        X_train_preprocessed = deepcopy(X_train)
        X_test_preprocessed = deepcopy(X_test)
        
        # Standardize
        X_train_preprocessed = self.scalerX.fit_transform(X_train_preprocessed)
        X_test_preprocessed = self.scalerX.transform(X_test_preprocessed)
        
        # Reshape for CNN input
        X_train_preprocessed = X_train_preprocessed[..., np.newaxis] if X_train_preprocessed.ndim == 2 else X_train_preprocessed
        X_test_preprocessed = X_test_preprocessed[..., np.newaxis] if X_test_preprocessed.ndim == 2 else X_test_preprocessed
        
        logger.info(f"Data preprocessed - Shape: {X_train_preprocessed.shape}")
        return X_train_preprocessed, X_test_preprocessed

    def preprocess_y(
        self,
        y_train: np.ndarray, 
        y_test: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Standardize and reshape y data in an immutable fashion.
        The standardization is alrady done but a reshape is necessary. 
        
        Args:
            y_train: Training targets
            y_test: Test targets
            
        Returns:
            Tuple of (y_train_processed, y_test_processed)
        """
        logger.info("Preprocessing y...")
        y_train_preprocessed = deepcopy(y_train)
        y_test_preprocessed = deepcopy(y_test)

        y_train_preprocessed = y_train_preprocessed[..., np.newaxis] if y_train_preprocessed.ndim == 1 else y_train_preprocessed
        y_test_preprocessed = y_test_preprocessed[..., np.newaxis] if y_test_preprocessed.ndim == 1 else y_test_preprocessed

        return y_train_preprocessed, y_test_preprocessed
    
    @abstractmethod
    def build_model(self) -> tf.keras.Model:
        """
        Build and compile the model. Must be implemented by subclass
        """
        pass
    
    def train(
        self, 
        X_train: np.ndarray, 
        y_train: np.ndarray,
        X_val: np.ndarray, 
        y_val: np.ndarray
    ) -> tf.keras.callbacks.History:
        """Train the model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            
        Returns:
            Training history object
        """
        logger.info("Starting model training...")
        
        if self.model is None:
            self.model = self.build_model()
        
        # Callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='loss', 
                patience=self.config.early_stopping_patience,
                restore_best_weights=True
            ),
            TqdmCallback(verbose=0)
        ]
        
        # Train
        history = self.model.fit(
            X_train, y_train,
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=0
        )
        
        logger.info(f"Training completed in {len(history.history['loss'])} epochs")
        return history
    
    def save_model(self) -> None:
        """Save the trained model to disk."""
        if self.model is None:
            raise ValueError("No model to save. Train or load a model first.")
        
        model_path = Path(self.config.model_path) / self.config.model_name
        model_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.model.save(str(model_path), overwrite=True, save_format='tf')
        logger.info(f"Model saved to {model_path}")
    
    def load_model(self) -> bool:
        """Load a saved model from disk.
        
        Returns:
            True if model was loaded, False if model doesn't exist
        """
        model_path = Path(self.config.model_path) / self.config.model_name
        
        if not model_path.exists():
            logger.info(f"Model not found at {model_path}")
            return False
        
        try:
            self.model = tf.keras.models.load_model(str(model_path))
            logger.info(f"Model loaded from {model_path}")
            return True
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def predict(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray
    ) -> Tuple[float, float, np.ndarray]:
        """Make predictions and evaluate model.
        
        Args:
            X_train: Training features
            X_test: Test features
            y_train: Training labels
            y_test: Test labels
            
        Returns:
            Tuple of (RMSEP, R2P, predictions)
        """
        if self.model is None:
            raise ValueError("No model available. Train or load a model first.")
        
        logger.info("Making predictions...")
        RMSEP, R2P, y_preds = cnn_prediction(
            self.model, X_train, X_test, y_train, y_test
        )
        
        logger.info(f"Predictions complete - RMSEP: {RMSEP:.4f}, R²: {R2P:.4f}")
        return RMSEP, R2P, y_preds
    
    def run_pipeline(self, force_retrain: bool = False) -> Dict[str, Any]:
        """Run the complete training or inference pipeline.
        
        Args:
            force_retrain: If True, retrain even if model exists
            
        Returns:
            Dictionary containing results and metrics
        """
        logger.info(f"TF v{tf.__version__} & Keras v{tf.keras.__version__}")
        
        # Load and preprocess data
        X_train, y_train, X_test, y_test = self.load_data()
        X_train_proc, X_test_proc = self.preprocess_X(X_train, X_test)
        y_train_proc, y_test_proc = self.preprocess_y(y_train, y_test)
        
        # Check if model exists
        model_exists = self.load_model()
        results = {}
        
        if not model_exists or force_retrain:
            # Train new model
            logger.info("Training new model...")
            history = self.train(X_train_proc, y_train_proc, X_test_proc, y_test_proc)
            
            # Save model
            self.save_model()
            
            # Plot training history
            plot_history(history)
            results['history'] = history.history
        else:
            logger.info("Using existing model for inference")
        
        # Make predictions
        RMSEP, R2P, y_preds = self.predict(
            X_train_proc, X_test_proc, y_train_proc, y_test_proc
        )
        
        # Display model summary
        self.model.summary()
        results.update({
            'RMSEP': RMSEP,
            'R2P': R2P,
            'predictions': y_preds
        })
        
        return results


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
    ipa_pipeline = ModelFactory.create_from_string(ModelType.IPA)
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
