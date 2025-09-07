import logging
import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any
from logging.handlers import RotatingFileHandler
from datetime import datetime
import json

class LoggerFactory:
    """
    Enterprise logging factory with structured logging, rotation, and filtering capabilities.
    Supports multiple output formats and configurable log levels per module.
    """
    
    _loggers: Dict[str, logging.Logger] = {}
    _initialized: bool = False
    _config: Dict[str, Any] = {}
    
    @classmethod
    def initialize(cls, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the logging system with configuration."""
        if cls._initialized:
            return
            
        cls._config = config or cls._get_default_config()
        cls._setup_logging_infrastructure()
        cls._initialized = True
    
    @classmethod
    def _get_default_config(cls) -> Dict[str, Any]:
        """Get default logging configuration."""
        return {
            'level': 'INFO',
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            'file_handler': {
                'enabled': True,
                'max_bytes': 10485760,  # 10MB
                'backup_count': 5
            },
            'console_handler': {
                'enabled': True
            }
        }
    
    @classmethod
    def _setup_logging_infrastructure(cls) -> None:
        """Setup logging directories and base configuration."""
        logs_dir = Path("logs")
        logs_dir.mkdir(exist_ok=True)
        
        # Setup root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, cls._config['level'].upper()))
        
        # Clear any existing handlers
        root_logger.handlers.clear()
        
        # Create formatters
        detailed_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
        simple_formatter = logging.Formatter(cls._config['format'])
        
        # Add console handler if enabled
        if cls._config.get('console_handler', {}).get('enabled', True):
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(simple_formatter)
            root_logger.addHandler(console_handler)
        
        # Add file handler if enabled
        if cls._config.get('file_handler', {}).get('enabled', True):
            file_handler = RotatingFileHandler(
                filename=logs_dir / "application.log",
                maxBytes=cls._config['file_handler']['max_bytes'],
                backupCount=cls._config['file_handler']['backup_count'],
                encoding='utf-8'
            )
            file_handler.setFormatter(detailed_formatter)
            root_logger.addHandler(file_handler)
    
    @classmethod
    def get_logger(cls, name: str, level: Optional[str] = None) -> logging.Logger:
        """
        Get or create a logger with specified name and configuration.
        
        Args:
            name: Logger name (typically __name__ or module name)
            level: Override log level for this specific logger
            
        Returns:
            Configured logger instance
        """
        if not cls._initialized:
            cls.initialize()
        
        if name in cls._loggers:
            return cls._loggers[name]
        
        logger = logging.getLogger(name)
        
        if level:
            logger.setLevel(getattr(logging, level.upper()))
        
        # Add structured logging capability
        logger = cls._enhance_logger(logger)
        
        cls._loggers[name] = logger
        return logger
    
    @classmethod
    def _enhance_logger(cls, logger: logging.Logger) -> logging.Logger:
        """Add structured logging methods to logger."""
        
        def log_structured(level: int, message: str, **kwargs) -> None:
            """Log structured data with additional context."""
            extra_data = {
                'timestamp': datetime.now().isoformat(),
                'context': kwargs
            }
            if kwargs:
                message = f"{message} | Context: {json.dumps(kwargs, default=str)}"
            logger.log(level, message, extra=extra_data)
        
        def debug_performance(func_name: str, execution_time: float, **kwargs) -> None:
            """Log performance metrics."""
            log_structured(
                logging.DEBUG,
                f"Performance: {func_name}",
                execution_time=execution_time,
                **kwargs
            )
        
        def log_model_metrics(epoch: int, metrics: Dict[str, float]) -> None:
            """Log model training metrics."""
            log_structured(
                logging.INFO,
                f"Training metrics - Epoch {epoch}",
                epoch=epoch,
                metrics=metrics
            )
        
        def log_data_validation(validation_result: Dict[str, Any]) -> None:
            """Log data validation results."""
            log_structured(
                logging.INFO,
                "Data validation completed",
                **validation_result
            )
        
        # Add enhanced methods to logger
        logger.log_structured = log_structured
        logger.debug_performance = debug_performance
        logger.log_model_metrics = log_model_metrics
        logger.log_data_validation = log_data_validation
        
        return logger

class PerformanceMonitor:
    """
    Performance monitoring and profiling utilities for model training
    and data processing operations.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or LoggerFactory.get_logger(self.__class__.__name__)
        self._timers = {}
    
    def __enter__(self):
        """Context manager entry."""
        self.start_time = datetime.now()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with automatic logging."""
        execution_time = (datetime.now() - self.start_time).total_seconds()
        if hasattr(self, 'operation_name'):
            self.logger.debug_performance(
                self.operation_name,
                execution_time,
                status='completed' if exc_type is None else 'failed'
            )
    
    def monitor_operation(self, operation_name: str):
        """Decorator for monitoring operation performance."""
        self.operation_name = operation_name
        return self
    
    def start_timer(self, timer_name: str) -> None:
        """Start a named timer."""
        self._timers[timer_name] = datetime.now()
    
    def end_timer(self, timer_name: str) -> float:
        """End a named timer and return elapsed time."""
        if timer_name not in self._timers:
            self.logger.warning(f"Timer '{timer_name}' was not started")
            return 0.0
        
        elapsed = (datetime.now() - self._timers[timer_name]).total_seconds()
        del self._timers[timer_name]
        
        self.logger.debug_performance(timer_name, elapsed)
        return elapsed

class MetricsCollector:
    """
    Centralized metrics collection system for monitoring model performance,
    resource usage, and system health during training and inference.
    """
    
    def __init__(self, save_to_file: bool = True):
        self.logger = LoggerFactory.get_logger(self.__class__.__name__)
        self.save_to_file = save_to_file
        self.metrics_history = []
        self.metrics_file = Path("logs") / "metrics.jsonl"
    
    def collect_training_metrics(self, epoch: int, fold: int, metrics: Dict[str, float]) -> None:
        """Collect and log training metrics."""
        metric_entry = {
            'timestamp': datetime.now().isoformat(),
            'type': 'training',
            'epoch': epoch,
            'fold': fold,
            **metrics
        }
        
        self.metrics_history.append(metric_entry)
        self.logger.log_model_metrics(epoch, metrics)
        
        if self.save_to_file:
            self._save_metric_to_file(metric_entry)
    
    def collect_system_metrics(self, **kwargs) -> None:
        """Collect system performance metrics."""
        try:
            import psutil
            cpu_percent = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            
            metric_entry = {
                'timestamp': datetime.now().isoformat(),
                'type': 'system',
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'memory_available_gb': memory.available / (1024**3),
                **kwargs
            }
            
            self.metrics_history.append(metric_entry)
            
            if self.save_to_file:
                self._save_metric_to_file(metric_entry)
                
        except ImportError:
            self.logger.debug("psutil not available for system metrics collection")
    
    def _save_metric_to_file(self, metric_entry: Dict[str, Any]) -> None:
        """Save metric entry to JSONL file."""
        try:
            with open(self.metrics_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(metric_entry) + '\n')
        except Exception as e:
            self.logger.error(f"Failed to save metrics to file: {e}")
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of collected metrics."""
        if not self.metrics_history:
            return {}
        
        training_metrics = [m for m in self.metrics_history if m.get('type') == 'training']
        if not training_metrics:
            return {}
        
        latest_metrics = training_metrics[-1]
        summary = {
            'total_entries': len(self.metrics_history),
            'training_entries': len(training_metrics),
            'latest_epoch': latest_metrics.get('epoch'),
            'latest_metrics': {k: v for k, v in latest_metrics.items() 
                             if k not in ['timestamp', 'type', 'epoch', 'fold']}
        }
        
        return summary