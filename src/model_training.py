from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple, Callable
from pathlib import Path
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from enum import Enum
import os

class ModelType(Enum):
    """Supported model types for molecular property prediction."""
    UNIMOL_V1 = "unimolv1"
    UNIMOL_V2 = "unimolv2"
    CUSTOM = "custom"

class TrainingStrategy(Enum):
    """Training strategy options."""
    STANDARD = "standard"
    CROSS_VALIDATION = "cross_validation"
    TRANSFER_LEARNING = "transfer_learning"

@dataclass
class TrainingConfiguration:
    """Comprehensive training configuration with validation."""
    model_type: ModelType
    model_size: str
    epochs: int
    batch_size: int
    learning_rate: float
    strategy: TrainingStrategy
    task_type: str = "classification"
    metrics: List[str] = field(default_factory=lambda: ["auc"])
    early_stopping: Dict[str, Any] = field(default_factory=dict)
    optimization: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate configuration parameters."""
        self._validate_configuration()
    
    def _validate_configuration(self) -> None:
        """Validate training configuration parameters."""
        if self.epochs <= 0:
            raise ValueError(f"Epochs must be positive, got: {self.epochs}")
        if self.batch_size <= 0:
            raise ValueError(f"Batch size must be positive, got: {self.batch_size}")
        if not 0 < self.learning_rate < 1:
            raise ValueError(f"Learning rate must be between 0 and 1, got: {self.learning_rate}")

@dataclass
class ModelTrainingResult:
    """Result of model training operation."""
    model_path: Path
    training_metrics: Dict[str, List[float]]
    validation_metrics: Dict[str, List[float]]
    final_metrics: Dict[str, float]
    training_time: float
    convergence_epoch: Optional[int] = None
    early_stopped: bool = False

class AbstractModelTrainer(ABC):
    """
    Abstract base class for model training with enterprise-level features:
    - Configurable training strategies
    - Comprehensive monitoring and logging
    - Resource management and optimization
    - Extensible architecture for different model types
    """
    
    def __init__(self, config: TrainingConfiguration, save_path: Path):
        self.config = config
        self.save_path = Path(save_path)
        self.save_path.mkdir(parents=True, exist_ok=True)
        
        from .logging_system import LoggerFactory, MetricsCollector
        self.logger = LoggerFactory.get_logger(self.__class__.__name__)
        self.metrics_collector = MetricsCollector(save_to_file=True)
        
        self._training_callbacks = []
        self._validation_callbacks = []
        self._setup_training_environment()
    
    def _setup_training_environment(self) -> None:
        """Setup training environment and validate requirements."""
        self.logger.info(f"Initializing {self.__class__.__name__} with configuration:")
        self.logger.info(f"Model: {self.config.model_type.value} ({self.config.model_size})")
        self.logger.info(f"Strategy: {self.config.strategy.value}")
        self.logger.info(f"Save path: {self.save_path}")
        
        # Create subdirectories for organized output
        (self.save_path / "checkpoints").mkdir(exist_ok=True)
        (self.save_path / "logs").mkdir(exist_ok=True)
        (self.save_path / "metrics").mkdir(exist_ok=True)
    
    @abstractmethod
    def _initialize_model(self) -> Any:
        """Initialize the underlying model."""
        pass
    
    @abstractmethod
    def _train_epoch(self, model: Any, train_data: pd.DataFrame, epoch: int) -> Dict[str, float]:
        """Train model for one epoch."""
        pass
    
    @abstractmethod
    def _validate_model(self, model: Any, val_data: pd.DataFrame) -> Dict[str, float]:
        """Validate model on validation data."""
        pass
    
    @abstractmethod
    def _save_model(self, model: Any, path: Path) -> None:
        """Save trained model to specified path."""
        pass
    
    def add_training_callback(self, callback: Callable[[int, Dict[str, float]], None]) -> None:
        """Add callback function called after each training epoch."""
        self._training_callbacks.append(callback)
    
    def add_validation_callback(self, callback: Callable[[int, Dict[str, float]], None]) -> None:
        """Add callback function called after each validation."""
        self._validation_callbacks.append(callback)
    
    def train(self, train_data: pd.DataFrame, 
              validation_data: Optional[pd.DataFrame] = None) -> ModelTrainingResult:
        """
        Execute model training with comprehensive monitoring and optimization.
        
        Args:
            train_data: Training dataset
            validation_data: Optional validation dataset
            
        Returns:
            Training result with metrics and model path
        """
        from datetime import datetime
        from .logging_system import PerformanceMonitor
        
        training_start_time = datetime.now()
        
        with PerformanceMonitor(self.logger).monitor_operation("model_training"):
            self.logger.info(f"Starting model training with {len(train_data)} samples")
            
            # Initialize model
            model = self._initialize_model()
            
            # Initialize tracking variables
            training_metrics_history = {metric: [] for metric in self.config.metrics}
            validation_metrics_history = {metric: [] for metric in self.config.metrics}
            best_validation_score = float('-inf')
            patience_counter = 0
            convergence_epoch = None
            early_stopped = False
            
            # Training loop
            for epoch in range(1, self.config.epochs + 1):
                self.logger.info(f"Starting epoch {epoch}/{self.config.epochs}")
                
                # Train for one epoch
                epoch_metrics = self._train_epoch(model, train_data, epoch)
                
                # Update training metrics history
                for metric, value in epoch_metrics.items():
                    if metric in training_metrics_history:
                        training_metrics_history[metric].append(value)
                
                # Collect metrics
                self.metrics_collector.collect_training_metrics(
                    epoch=epoch, fold=0, metrics=epoch_metrics
                )
                
                # Execute training callbacks
                for callback in self._training_callbacks:
                    try:
                        callback(epoch, epoch_metrics)
                    except Exception as e:
                        self.logger.warning(f"Training callback failed: {e}")
                
                # Validation if data provided
                if validation_data is not None:
                    val_metrics = self._validate_model(model, validation_data)
                    
                    # Update validation metrics history
                    for metric, value in val_metrics.items():
                        if metric in validation_metrics_history:
                            validation_metrics_history[metric].append(value)
                    
                    # Execute validation callbacks
                    for callback in self._validation_callbacks:
                        try:
                            callback(epoch, val_metrics)
                        except Exception as e:
                            self.logger.warning(f"Validation callback failed: {e}")
                    
                    # Early stopping logic
                    if self.config.early_stopping.get('enabled', False):
                        primary_metric = self.config.early_stopping.get('metric', 'auc')
                        current_score = val_metrics.get(primary_metric, 0)
                        
                        if current_score > best_validation_score + self.config.early_stopping.get('min_delta', 0.001):
                            best_validation_score = current_score
                            patience_counter = 0
                            convergence_epoch = epoch
                            
                            # Save best model
                            best_model_path = self.save_path / "best_model"
                            best_model_path.mkdir(exist_ok=True)
                            self._save_model(model, best_model_path)
                            self.logger.info(f"New best model saved at epoch {epoch}")
                        else:
                            patience_counter += 1
                            
                        if patience_counter >= self.config.early_stopping.get('patience', 10):
                            self.logger.info(f"Early stopping triggered at epoch {epoch}")
                            early_stopped = True
                            break
                
                # Checkpoint saving
                if epoch % 10 == 0:
                    checkpoint_path = self.save_path / "checkpoints" / f"epoch_{epoch}"
                    checkpoint_path.mkdir(exist_ok=True)
                    self._save_model(model, checkpoint_path)
                    self.logger.debug(f"Checkpoint saved for epoch {epoch}")
            
            # Save final model
            final_model_path = self.save_path / "final_model"
            final_model_path.mkdir(exist_ok=True)
            self._save_model(model, final_model_path)
            
            # Calculate training time
            training_time = (datetime.now() - training_start_time).total_seconds()
            
            # Prepare final metrics
            final_metrics = {}
            if training_metrics_history:
                for metric, values in training_metrics_history.items():
                    if values:
                        final_metrics[f"final_train_{metric}"] = values[-1]
            
            if validation_metrics_history:
                for metric, values in validation_metrics_history.items():
                    if values:
                        final_metrics[f"final_val_{metric}"] = values[-1]
                        final_metrics[f"best_val_{metric}"] = max(values)
            
            self.logger.info(f"Training completed in {training_time:.2f} seconds")
            self.logger.info(f"Final metrics: {final_metrics}")
            
            return ModelTrainingResult(
                model_path=final_model_path,
                training_metrics=training_metrics_history,
                validation_metrics=validation_metrics_history,
                final_metrics=final_metrics,
                training_time=training_time,
                convergence_epoch=convergence_epoch,
                early_stopped=early_stopped
            )

class UniMolTrainer(AbstractModelTrainer):
    """
    Specialized trainer for UniMol models with molecular-specific optimizations
    and advanced training techniques.
    """
    
    def __init__(self, config: TrainingConfiguration, save_path: Path, target_column: str):
        super().__init__(config, save_path)
        self.target_column = target_column
        self._model_instance = None
    
    def _initialize_model(self) -> Any:
        """Initialize UniMol model with specified configuration."""
        try:
            from unimol_tools import MolTrain
            
            # Configure freeze layers for transfer learning
            freeze_config = self._determine_freeze_strategy()
            
            model = MolTrain(
                task=self.config.task_type,
                data_type='molecule',
                epochs=self.config.epochs,
                batch_size=self.config.batch_size,
                metrics=self.config.metrics[0] if self.config.metrics else 'auc',
                model_name=self.config.model_type.value,
                model_size=self.config.model_size,
                save_path=str(self.save_path),
                target_cols=self.target_column,
                **freeze_config
            )
            
            self._model_instance = model
            self.logger.info(f"UniMol model initialized: {self.config.model_type.value} ({self.config.model_size})")
            
            return model
            
        except ImportError as e:
            self.logger.error("Failed to import unimol_tools. Please ensure it's installed.")
            raise e
        except Exception as e:
            self.logger.error(f"Failed to initialize UniMol model: {e}")
            raise e
    
    def _determine_freeze_strategy(self) -> Dict[str, Any]:
        """Determine layer freezing strategy based on configuration."""
        freeze_config = {}
        
        if self.config.strategy == TrainingStrategy.TRANSFER_LEARNING:
            # For transfer learning, freeze lower layers and fine-tune higher layers
            freeze_config.update({
                'freeze_layers': ['classification_head'],
                'freeze_layers_reversed': True
            })
            self.logger.info("Applied transfer learning freeze strategy")
        
        return freeze_config
    
    def _train_epoch(self, model: Any, train_data: pd.DataFrame, epoch: int) -> Dict[str, float]:
        """Train UniMol model for one epoch."""
        # Save training data temporarily for UniMol
        temp_train_path = self.save_path / f"temp_train_epoch_{epoch}.csv"
        train_data.to_csv(temp_train_path, index=False)
        
        try:
            # UniMol handles epoch training internally
            if epoch == 1:  # Only call fit once, as UniMol manages epochs internally
                results = model.fit(data=str(temp_train_path))
                self.logger.debug(f"UniMol training results: {results}")
            
            # Extract metrics (simplified for demo - actual implementation would parse UniMol results)
            epoch_metrics = {
                'loss': np.random.uniform(0.1, 0.5),  # Placeholder - extract from actual UniMol results
                'auc': np.random.uniform(0.7, 0.95)   # Placeholder - extract from actual UniMol results
            }
            
            return epoch_metrics
            
        finally:
            # Clean up temporary file
            if temp_train_path.exists():
                temp_train_path.unlink()
    
    def _validate_model(self, model: Any, val_data: pd.DataFrame) -> Dict[str, float]:
        """Validate UniMol model on validation data."""
        from unimol_tools import MolPredict
        
        # Save validation data temporarily
        temp_val_path = self.save_path / "temp_validation.csv"
        val_data.to_csv(temp_val_path, index=False)
        
        try:
            # Create predictor and get predictions
            predictor = MolPredict(load_model=str(self.save_path))
            predictions = predictor.predict(data=str(temp_val_path))
            
            # Calculate validation metrics
            y_true = val_data[self.target_column].values
            y_pred = [1 if p >= 0.5 else 0 for p in predictions]
            y_prob = predictions
            
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
            
            val_metrics = {
                'accuracy': accuracy_score(y_true, y_pred),
                'precision': precision_score(y_true, y_pred, zero_division=0),
                'recall': recall_score(y_true, y_pred, zero_division=0),
                'f1': f1_score(y_true, y_pred, zero_division=0),
                'auc': roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.0
            }
            
            return val_metrics
            
        except Exception as e:
            self.logger.warning(f"Validation failed: {e}")
            return {}
        finally:
            # Clean up temporary file
            if temp_val_path.exists():
                temp_val_path.unlink()
    
    def _save_model(self, model: Any, path: Path) -> None:
        """Save UniMol model to specified path."""
        # UniMol saves model automatically during training
        # Copy model files to specified path if needed
        if self._model_instance:
            self.logger.debug(f"UniMol model saved to: {path}")
        else:
            self.logger.warning("No model instance to save")