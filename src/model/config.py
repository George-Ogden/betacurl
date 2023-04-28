from tensorflow.keras import callbacks, optimizers
from wandb.keras import WandbCallback
from tqdm.keras import TqdmCallback

from typing import Any, ClassVar, Dict, List, Optional
from dataclasses import dataclass, field

from ..utils import Config

@dataclass
class ModelConfig(Config):
    output_activation: str = "linear"

@dataclass
class MLPModelConfig(ModelConfig):
    hidden_size: int = 256
    dropout: float = .1

@dataclass
class FCNNConfig(MLPModelConfig):
    hidden_layers: int = 3
    def __post_init__(self):
        assert self.hidden_layers >= 1

@dataclass
class TrainingConfig(Config):
    training_epochs: int = 10
    """number of epochs to train each model for"""
    batch_size: int = 256
    """training batch size"""
    training_patience: Optional[int] = 7
    """number of epochs without improvement during training (0 to ignore)"""
    lr: float = 1e-3
    """model learning rate"""
    validation_split: float = 0.1
    """proportion of data to validate on"""
    loss: str = "mse"
    optimizer_type: str = "Adam"
    metrics: List[str] = field(default_factory=lambda: ["mae"])
    additional_callbacks: Optional[List[callbacks.Callback]] = None
    compile_kwargs: Optional[Dict[str, Any]] = None
    fit_kwargs: Optional[Dict[str, Any]] = None
    optimizer_kwargs: Optional[Dict[str, Any]] = field(default_factory=lambda:{"clipnorm": None})
    verbose: ClassVar[int] = 0

    @property
    def optimizer(self) -> optimizers.Optimizer:
        return type(optimizers.get(self.optimizer_type))(
            learning_rate=self.lr,
            **(self.optimizer_kwargs or {})
        )
    
    @property
    def callbacks(self) -> List[callbacks.Callback]:
        return [
            WandbCallback(),
            TqdmCallback(desc="Training"),
            callbacks.TerminateOnNaN(),
        ] + (
            self.additional_callbacks or []
        ) + (
            [
                callbacks.EarlyStopping(
                    patience=self.training_patience,
                    monitor="val_" + (self.metrics[0] if len(self.metrics) > 0 else "loss"),
                    restore_best_weights=True
                )
            ] if self.training_patience is not None else []
        )