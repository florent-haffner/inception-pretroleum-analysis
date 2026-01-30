from dataclasses import dataclass

from src.model_utils.model_type import ModelType


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
