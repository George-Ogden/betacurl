from tensorflow.keras import callbacks, optimizers
from wandb.keras import WandbCallback
from tqdm.keras import TqdmCallback

from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field

@dataclass
class ModelConfig:
    output_activation:  str = "sigmoid"

@dataclass
class SimpleLinearModelConfig(ModelConfig):
    hidden_size: int = 128

@dataclass
class FCNNConfig(SimpleLinearModelConfig):
    dropout: float = .1

@dataclass
class TrainingConfig:
    training_epochs: int = 20
    """number of epochs to train each model for"""
    batch_size: int = 64
    """training batch size"""
    training_patience: int = 7
    """number of epochs without improvement during training (0 to ignore)"""
    lr: float = 1e-2
    """model learning rate"""
    validation_split: float = 0.1
    """proportion of data to validate on"""
    loss: str = "mse"
    optimizer_type: str = "Adam"
    metrics: List[str] = field(default_factory=lambda: ["mae"])
    additional_callbacks: Optional[List[callbacks.Callback]] = None
    compile_kwargs: Optional[Dict[str, Any]] = None
    fit_kwargs: Optional[Dict[str, Any]] = None

    @property
    def optimizer(self) -> optimizers.Optimizer:
        return type(optimizers.get(self.optimizer_type))(
            learning_rate=self.lr
        )
    
    @property
    def callbacks(self) -> List[callbacks.Callback]:
        return [
            WandbCallback(),
            TqdmCallback(desc="Training"),
        ] + (
            self.additional_callbacks or []
        ) + (
            [
                callbacks.EarlyStopping(
                    patience=self.training_patience,
                    monitor="val_mae",
                    restore_best_weights=True
                )
            ] if self.training_patience > 0 else []
        )