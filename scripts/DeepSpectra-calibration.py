#!/usr/bin/env python
# coding: utf-8
import os
import logging
from copy import deepcopy
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass
import warnings

import numpy as np
from scipy.io import loadmat
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import optimizers
from tqdm.keras import TqdmCallback

from python.DeepSpectra_architecture import DeepSpectra
from python.plot_utils import plot_history, plot_val, plot_err
from python.regression_utils import cnn_prediction

warnings.filterwarnings("ignore")
sns.set_theme(style="ticks", palette='muted')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelConfig:
    """Configuration for DeepSpectra model training and inference."""
    # Paths
    data_path: str = "data/raw-kennard-reduced_range.mat"
    # Alternative: Use full range dataset
    # data_path = "data/raw-kennard-full_range.mat"

    y_train_path: str = "data/y_train_scaled.csv"
    y_test_path: str = "data/y_test_scaled.csv"
    model_path: str = "model/"
    model_name: str = "DeepSpectra"
    
    # Training hyperparameters
    learning_rate: float = 0.01
    batch_size: int = 32
    epochs: int = 500
    
    # Regularization
    regularization_coef: float = 0.001
    dropout_rate: float = 0.2
    
    # Other settings
    seed_value: int = 1
    early_stopping_patience: int = 50
    decay_steps: int = 10000
    decay_rate: float = 0.001
    loss_function: str = 'mse'


class DeepSpectraModelPipeline:
    """End-to-end pipeline for DeepSpectra model training and inference."""
    
    def __init__(self, config: ModelConfig):
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
    
    def build_model(self) -> tf.keras.Model:
        """Build and compile the DeepSpectra model.
        
        Returns:
            Compiled Keras model
        """
        logger.info("Building DeepSpectra model...")
        
        model = DeepSpectra(
            seed_value=self.config.seed_value,
            regularization_factor=self.config.regularization_coef,
            dropout_rate=self.config.dropout_rate
        )
        
        # Learning rate schedule
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=self.config.learning_rate,
            decay_steps=self.config.decay_steps,
            decay_rate=self.config.decay_rate
        )
        
        optimizer = optimizers.Adam(learning_rate=lr_schedule)
        model.compile(
            optimizer=optimizer, 
            loss=self.config.loss_function, 
            metrics=['mse']
        )
        
        logger.info("Model built and compiled")
        return model
    
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


def main():
    """Main execution function."""
    # Create configuration
    config = ModelConfig()
    
    # Initialize pipeline
    pipeline = DeepSpectraModelPipeline(config)
    results = pipeline.run_pipeline(force_retrain=False)
    
    logger.info("Pipeline execution completed successfully")
    logger.info(f"Final metrics - RMSEP: {results['RMSEP']:.4f}, R²: {results['R2P']:.4f}")


if __name__ == "__main__":
    main()
