#!/usr/bin/env python3
"""
UC Drug Activity Predictor - Enterprise Machine Learning Pipeline
================================================================

A production-ready molecular property prediction system built on UniMol
with comprehensive data validation, model training orchestration,
and enterprise-grade monitoring capabilities.

Author: ML Engineering Team
Version: 2.1.0
License: MIT
"""

import sys
import os
from pathlib import Path

# Add src to Python path for modular imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Import our enterprise modules
from config_manager import ConfigurationManager, get_config, get_config_value
from logging_system import LoggerFactory, PerformanceMonitor, MetricsCollector
from data_processing import MolecularDataProcessor, ValidationReport
from model_training import (
    UniMolTrainer, TrainingConfiguration, ModelType, 
    TrainingStrategy, ModelTrainingResult
)

class DrugActivityPredictionOrchestrator:
    """
    Main orchestration class for the drug activity prediction pipeline.
    
    This class coordinates all aspects of the machine learning workflow:
    - Configuration management and validation
    - Data loading and preprocessing
    - Model training and evaluation
    - Results aggregation and reporting
    - Resource management and cleanup
    """
    
    def __init__(self, config_override: Optional[Dict[str, Any]] = None):
        """
        Initialize the prediction orchestrator with configuration management.
        
        Args:
            config_override: Optional configuration overrides
        """
        self._setup_configuration(config_override)
        self._initialize_logging()
        self._initialize_components()
        self._setup_environment()
    
    def _setup_configuration(self, config_override: Optional[Dict[str, Any]]) -> None:
        """Setup and validate configuration management."""
        self.config_manager = ConfigurationManager()
        self.config = self.config_manager.load_configuration('settings')
        self.runtime_config = self.config_manager.load_configuration('runtime')
        
        if config_override:
            self.config.update(config_override)
    
    def _initialize_logging(self) -> None:
        """Initialize enterprise logging system."""
        LoggerFactory.initialize(self.config.get('logging', {}))
        self.logger = LoggerFactory.get_logger(__name__)
        self.metrics_collector = MetricsCollector(
            save_to_file=get_config_value('monitoring.save_training_history', default=True)
        )
        
        self.logger.info("="*80)
        self.logger.info(f"Initializing {self.config['app']['name']} v{self.config['app']['version']}")
        self.logger.info("="*80)
    
    def _initialize_components(self) -> None:
        """Initialize core pipeline components."""
        self.data_processor = MolecularDataProcessor(self.config['data'])
        self.results_aggregator = ModelResultsAggregator(self.logger)
        
        # Setup reproducibility
        random_seed = get_config_value('training.cross_validation.random_state', default=42)
        np.random.seed(random_seed)
        self.logger.info(f"Random seed set to {random_seed} for reproducibility")
    
    def _setup_environment(self) -> None:
        """Setup working environment and directories."""
        self.results_dir = Path(get_config_value('paths.results_dir', default='immunosuppressant_results'))
        self.results_dir.mkdir(exist_ok=True)
        
        # Setup environment variables from runtime config
        env_vars = self.runtime_config.get('runtime', {}).get('environment_variables', {})
        for key, value in env_vars.items():
            os.environ[key] = str(value)
            self.logger.debug(f"Set environment variable: {key}")
    
    def execute_full_pipeline(self) -> Dict[str, Any]:
        """
        Execute the complete drug activity prediction pipeline.
        
        Returns:
            Dictionary containing comprehensive results and metrics
        """
        with PerformanceMonitor(self.logger).monitor_operation("full_pipeline_execution"):
            
            # Step 1: Data Loading and Validation
            self.logger.info("Phase 1: Data Loading and Preprocessing")
            data, validation_report = self._load_and_validate_data()
            
            # Step 2: Cross-validation Training
            cv_results = None
            if get_config_value('training.cross_validation.n_folds', default=0) > 1:
                self.logger.info("Phase 2: Cross-validation Training")
                cv_results = self._execute_cross_validation(data)
            
            # Step 3: Final Model Training
            self.logger.info("Phase 3: Final Model Training")
            final_model_result = self._train_final_model(data)
            
            # Step 4: Model Evaluation and Prediction
            self.logger.info("Phase 4: Model Evaluation")
            evaluation_results = self._evaluate_final_model(final_model_result)
            
            # Step 5: Results Compilation
            self.logger.info("Phase 5: Results Compilation")
            comprehensive_results = self._compile_final_results(
                validation_report, cv_results, final_model_result, evaluation_results
            )
            
            self.logger.info("Pipeline execution completed successfully")
            return comprehensive_results
    
    def _load_and_validate_data(self) -> Tuple[pd.DataFrame, ValidationReport]:
        """Load and validate input data with comprehensive checking."""
        data_file = get_config_value('data.input_file', default='data.csv')
        
        self.logger.info(f"Loading data from: {data_file}")
        data, validation_report = self.data_processor.process_data_pipeline(data_file)
        
        self.logger.info(f"Data validation completed:")
        self.logger.info(f"  - Status: {validation_report.status.value}")
        self.logger.info(f"  - Total records: {validation_report.total_records}")
        self.logger.info(f"  - Valid records: {validation_report.valid_records}")
        self.logger.info(f"  - Warnings: {len(validation_report.warnings)}")
        self.logger.info(f"  - Errors: {len(validation_report.errors)}")
        
        return data, validation_report
    
    def _execute_cross_validation(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Execute cross-validation training with comprehensive metrics tracking."""
        cv_config = self.config['training']['cross_validation']
        n_folds = cv_config['n_folds']
        
        kf = KFold(
            n_splits=n_folds,
            shuffle=cv_config['shuffle'],
            random_state=cv_config['random_state']
        )
        
        cv_results = {
            'fold_results': [],
            'aggregated_metrics': {},
            'fold_models': []
        }
        
        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(data), 1):
            self.logger.info(f"Training fold {fold_idx}/{n_folds}")
            
            with PerformanceMonitor(self.logger).monitor_operation(f"fold_{fold_idx}_training"):
                fold_result = self._train_single_fold(
                    data.iloc[train_idx].reset_index(drop=True),
                    data.iloc[val_idx].reset_index(drop=True),
                    fold_idx
                )
                cv_results['fold_results'].append(fold_result)
                cv_results['fold_models'].append(fold_result.model_path)
        
        # Aggregate cross-validation metrics
        cv_results['aggregated_metrics'] = self.results_aggregator.aggregate_cv_results(
            cv_results['fold_results']
        )
        
        self._log_cv_summary(cv_results['aggregated_metrics'])
        return cv_results
    
    def _train_single_fold(self, train_data: pd.DataFrame, 
                          val_data: pd.DataFrame, fold_idx: int) -> ModelTrainingResult:
        """Train model for a single cross-validation fold."""
        fold_dir = self.results_dir / f'fold_{fold_idx}'
        fold_dir.mkdir(exist_ok=True)
        
        # Save fold data for reference and debugging
        train_data.to_csv(fold_dir / 'train_data.csv', index=False)
        val_data.to_csv(fold_dir / 'val_data.csv', index=False)
        
        # Create training configuration
        training_config = self._create_training_configuration()
        
        # Initialize trainer
        trainer = UniMolTrainer(
            config=training_config,
            save_path=fold_dir,
            target_column=get_config_value('data.target_column', default='TARGET')
        )
        
        # Add monitoring callbacks
        trainer.add_training_callback(
            lambda epoch, metrics: self.metrics_collector.collect_training_metrics(
                epoch, fold_idx, metrics
            )
        )
        
        # Execute training
        result = trainer.train(train_data, val_data)
        
        self.logger.info(f"Fold {fold_idx} completed:")
        for metric_name, value in result.final_metrics.items():
            self.logger.info(f"  - {metric_name}: {value:.4f}")
        
        return result
    
    def _train_final_model(self, data: pd.DataFrame) -> ModelTrainingResult:
        """Train final model on entire dataset."""
        final_model_dir = self.results_dir / 'final_model'
        final_model_dir.mkdir(exist_ok=True)
        
        self.logger.info(f"Training final model on {len(data)} samples")
        
        # Create training configuration with transfer learning strategy
        training_config = self._create_training_configuration(
            strategy=TrainingStrategy.TRANSFER_LEARNING
        )
        
        # Initialize trainer
        trainer = UniMolTrainer(
            config=training_config,
            save_path=final_model_dir,
            target_column=get_config_value('data.target_column', default='TARGET')
        )
        
        # Execute training
        result = trainer.train(data)
        
        self.logger.info("Final model training completed:")
        for metric_name, value in result.final_metrics.items():
            self.logger.info(f"  - {metric_name}: {value:.4f}")
        
        return result
    
    def _evaluate_final_model(self, model_result: ModelTrainingResult) -> Dict[str, Any]:
        """Evaluate final model on test data if available."""
        test_file = get_config_value('data.test_file')
        
        if not test_file or not Path(test_file).exists():
            self.logger.info("No test file available for final evaluation")
            return {}
        
        self.logger.info(f"Evaluating final model on test data: {test_file}")
        
        try:
            from unimol_tools import MolPredict
            
            # Load test data
            test_data = pd.read_csv(test_file)
            self.logger.info(f"Loaded {len(test_data)} test samples")
            
            # Make predictions
            predictor = MolPredict(load_model=str(model_result.model_path))
            predictions = predictor.predict(data=test_file)
            
            # Save predictions
            prediction_results = pd.DataFrame({
                'SMILES': test_data.get('SMILES', range(len(predictions))),
                'predicted_probability': predictions,
                'predicted_class': [1 if p >= 0.5 else 0 for p in predictions]
            })
            
            prediction_file = self.results_dir / 'final_predictions.csv'
            prediction_results.to_csv(prediction_file, index=False)
            
            self.logger.info(f"Predictions saved to: {prediction_file}")
            self.logger.info(f"Prediction summary:")
            self.logger.info(f"  - Positive predictions: {sum(prediction_results['predicted_class'])}")
            self.logger.info(f"  - Negative predictions: {len(prediction_results) - sum(prediction_results['predicted_class'])}")
            
            return {
                'predictions_file': str(prediction_file),
                'num_predictions': len(predictions),
                'positive_predictions': sum(prediction_results['predicted_class']),
                'prediction_distribution': prediction_results['predicted_probability'].describe().to_dict()
            }
            
        except Exception as e:
            self.logger.error(f"Failed to evaluate final model: {e}")
            return {'error': str(e)}
    
    def _create_training_configuration(self, 
                                     strategy: TrainingStrategy = TrainingStrategy.STANDARD) -> TrainingConfiguration:
        """Create training configuration from loaded config."""
        model_config = self.config['model']
        training_config = self.config['training']
        
        return TrainingConfiguration(
            model_type=ModelType(model_config['base_model']),
            model_size=model_config['size'],
            epochs=training_config['epochs'],
            batch_size=training_config['batch_size'],
            learning_rate=training_config.get('learning_rate', 0.001),
            strategy=strategy,
            task_type=model_config['task_type'],
            metrics=training_config['metrics'],
            early_stopping=training_config.get('early_stopping', {}),
        )
    
    def _log_cv_summary(self, aggregated_metrics: Dict[str, Any]) -> None:
        """Log cross-validation summary statistics."""
        self.logger.info("Cross-validation Results Summary:")
        self.logger.info("="*50)
        
        for metric_name, stats in aggregated_metrics.items():
            if isinstance(stats, dict):
                self.logger.info(f"{metric_name}:")
                self.logger.info(f"  Mean: {stats.get('mean', 0):.4f}")
                self.logger.info(f"  Std:  {stats.get('std', 0):.4f}")
                self.logger.info(f"  Min:  {stats.get('min', 0):.4f}")
                self.logger.info(f"  Max:  {stats.get('max', 0):.4f}")
    
    def _compile_final_results(self, validation_report: ValidationReport,
                             cv_results: Optional[Dict[str, Any]],
                             final_model_result: ModelTrainingResult,
                             evaluation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Compile comprehensive results from all pipeline phases."""
        results = {
            'pipeline_info': {
                'version': self.config['app']['version'],
                'configuration': self.config,
                'execution_timestamp': pd.Timestamp.now().isoformat()
            },
            'data_validation': {
                'status': validation_report.status.value,
                'total_records': validation_report.total_records,
                'valid_records': validation_report.valid_records,
                'warnings_count': len(validation_report.warnings),
                'errors_count': len(validation_report.errors),
                'processing_time': validation_report.processing_time
            },
            'final_model': {
                'model_path': str(final_model_result.model_path),
                'training_time': final_model_result.training_time,
                'final_metrics': final_model_result.final_metrics,
                'convergence_epoch': final_model_result.convergence_epoch,
                'early_stopped': final_model_result.early_stopped
            },
            'evaluation_results': evaluation_results,
            'system_metrics': self.metrics_collector.get_metrics_summary()
        }
        
        if cv_results:
            results['cross_validation'] = {
                'n_folds': len(cv_results['fold_results']),
                'aggregated_metrics': cv_results['aggregated_metrics']
            }
        
        # Save comprehensive results
        results_file = self.results_dir / 'comprehensive_results.json'
        import json
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, default=str)
        
        self.logger.info(f"Comprehensive results saved to: {results_file}")
        return results

class ModelResultsAggregator:
    """Utility class for aggregating and analyzing model training results."""
    
    def __init__(self, logger):
        self.logger = logger
    
    def aggregate_cv_results(self, fold_results: List[ModelTrainingResult]) -> Dict[str, Dict[str, float]]:
        """Aggregate cross-validation results across all folds."""
        if not fold_results:
            return {}
        
        # Collect all metrics
        all_metrics = {}
        for result in fold_results:
            for metric_name, value in result.final_metrics.items():
                if metric_name not in all_metrics:
                    all_metrics[metric_name] = []
                all_metrics[metric_name].append(value)
        
        # Calculate statistics
        aggregated = {}
        for metric_name, values in all_metrics.items():
            values_array = np.array(values)
            aggregated[metric_name] = {
                'mean': float(np.mean(values_array)),
                'std': float(np.std(values_array)),
                'min': float(np.min(values_array)),
                'max': float(np.max(values_array)),
                'median': float(np.median(values_array))
            }
        
        return aggregated

def main():
    """Main entry point for the drug activity prediction pipeline."""
    try:
        # Initialize orchestrator
        orchestrator = DrugActivityPredictionOrchestrator()
        
        # Execute full pipeline
        results = orchestrator.execute_full_pipeline()
        
        print("\\n" + "="*80)
        print("PIPELINE EXECUTION COMPLETED SUCCESSFULLY")
        print("="*80)
        print(f"Results saved to: {orchestrator.results_dir}")
        
        # Print key metrics if available
        if 'final_model' in results and 'final_metrics' in results['final_model']:
            print("\\nFinal Model Performance:")
            for metric, value in results['final_model']['final_metrics'].items():
                print(f"  {metric}: {value:.4f}")
        
        return 0
        
    except Exception as e:
        print(f"Pipeline execution failed: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()