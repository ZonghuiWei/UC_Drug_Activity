from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple, Union
from pathlib import Path
import pandas as pd
import numpy as np
from dataclasses import dataclass
from enum import Enum

class DataValidationResult(Enum):
    """Enumeration of data validation results."""
    VALID = "valid"
    WARNING = "warning"
    ERROR = "error"

@dataclass
class ValidationReport:
    """Data validation report with detailed results."""
    status: DataValidationResult
    total_records: int
    valid_records: int
    invalid_records: int
    warnings: List[str]
    errors: List[str]
    processing_time: float

class AbstractDataProcessor(ABC):
    """
    Abstract base class for data processing operations with comprehensive
    validation, transformation, and quality assurance capabilities.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        from .logging_system import LoggerFactory
        self.logger = LoggerFactory.get_logger(self.__class__.__name__)
        self._validation_rules = self._initialize_validation_rules()
    
    @abstractmethod
    def _initialize_validation_rules(self) -> Dict[str, Any]:
        """Initialize validation rules specific to processor type."""
        pass
    
    @abstractmethod
    def load_data(self, data_path: Union[str, Path]) -> pd.DataFrame:
        """Load data from specified path."""
        pass
    
    @abstractmethod
    def validate_data(self, data: pd.DataFrame) -> ValidationReport:
        """Validate data against predefined rules."""
        pass
    
    @abstractmethod
    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply preprocessing transformations to data."""
        pass
    
    def process_data_pipeline(self, data_path: Union[str, Path]) -> Tuple[pd.DataFrame, ValidationReport]:
        """
        Execute complete data processing pipeline with validation and preprocessing.
        
        Args:
            data_path: Path to input data file
            
        Returns:
            Tuple of processed data and validation report
        """
        from .logging_system import PerformanceMonitor
        
        with PerformanceMonitor(self.logger).monitor_operation("data_processing_pipeline"):
            # Load data
            self.logger.info(f"Loading data from: {data_path}")
            raw_data = self.load_data(data_path)
            
            # Validate data
            self.logger.info("Validating data quality and structure")
            validation_report = self.validate_data(raw_data)
            
            if validation_report.status == DataValidationResult.ERROR:
                raise ValueError(f"Data validation failed: {validation_report.errors}")
            
            # Log validation results
            self.logger.log_data_validation({
                'status': validation_report.status.value,
                'total_records': validation_report.total_records,
                'valid_records': validation_report.valid_records,
                'warnings_count': len(validation_report.warnings),
                'errors_count': len(validation_report.errors)
            })
            
            # Preprocess data
            self.logger.info("Applying data preprocessing transformations")
            processed_data = self.preprocess_data(raw_data)
            
            self.logger.info(f"Data processing completed. Final dataset size: {len(processed_data)}")
            
            return processed_data, validation_report

class MolecularDataProcessor(AbstractDataProcessor):
    """
    Specialized data processor for molecular SMILES data with chemistry-specific
    validation and preprocessing capabilities.
    """
    
    def _initialize_validation_rules(self) -> Dict[str, Any]:
        """Initialize molecular data validation rules."""
        return {
            'required_columns': [
                self.config.get('smiles_column', 'SMILES'),
                self.config.get('target_column', 'TARGET')
            ],
            'smiles_validation': {
                'enabled': self.config.get('preprocessing', {}).get('validate_smiles', True),
                'min_length': 3,
                'max_length': 1000
            },
            'target_validation': {
                'allowed_values': [0, 1] if self.config.get('model', {}).get('task_type') == 'classification' else None
            },
            'duplicate_detection': {
                'enabled': self.config.get('preprocessing', {}).get('remove_duplicates', True)
            }
        }
    
    def load_data(self, data_path: Union[str, Path]) -> pd.DataFrame:
        """Load molecular data from CSV or Excel files."""
        data_path = Path(data_path)
        
        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")
        
        file_extension = data_path.suffix.lower()
        
        try:
            if file_extension == '.csv':
                data = pd.read_csv(data_path, encoding='utf-8')
            elif file_extension in ['.xlsx', '.xls']:
                data = pd.read_excel(data_path)
            else:
                raise ValueError(f"Unsupported file format: {file_extension}")
            
            self.logger.info(f"Successfully loaded {len(data)} records from {data_path}")
            return data
            
        except Exception as e:
            self.logger.error(f"Failed to load data from {data_path}: {str(e)}")
            raise
    
    def validate_data(self, data: pd.DataFrame) -> ValidationReport:
        """Comprehensive validation of molecular data."""
        from datetime import datetime
        start_time = datetime.now()
        
        warnings = []
        errors = []
        
        # Check required columns
        required_columns = self._validation_rules['required_columns']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            errors.append(f"Missing required columns: {missing_columns}")
        
        # Validate SMILES strings if column exists
        smiles_col = self.config.get('smiles_column', 'SMILES')
        if smiles_col in data.columns:
            smiles_validation = self._validate_smiles_column(data[smiles_col])
            warnings.extend(smiles_validation['warnings'])
            errors.extend(smiles_validation['errors'])
        
        # Validate target column if exists
        target_col = self.config.get('target_column', 'TARGET')
        if target_col in data.columns:
            target_validation = self._validate_target_column(data[target_col])
            warnings.extend(target_validation['warnings'])
            errors.extend(target_validation['errors'])
        
        # Check for duplicates
        if self._validation_rules['duplicate_detection']['enabled']:
            duplicate_count = data.duplicated().sum()
            if duplicate_count > 0:
                warnings.append(f"Found {duplicate_count} duplicate records")
        
        # Determine overall validation status
        if errors:
            status = DataValidationResult.ERROR
        elif warnings:
            status = DataValidationResult.WARNING
        else:
            status = DataValidationResult.VALID
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return ValidationReport(
            status=status,
            total_records=len(data),
            valid_records=len(data) - len([w for w in warnings if 'invalid' in w.lower()]),
            invalid_records=len([w for w in warnings if 'invalid' in w.lower()]),
            warnings=warnings,
            errors=errors,
            processing_time=processing_time
        )
    
    def _validate_smiles_column(self, smiles_series: pd.Series) -> Dict[str, List[str]]:
        """Validate SMILES strings for basic structural correctness."""
        warnings = []
        errors = []
        
        # Check for null values
        null_count = smiles_series.isnull().sum()
        if null_count > 0:
            errors.append(f"Found {null_count} null SMILES values")
        
        # Check SMILES length
        smiles_config = self._validation_rules['smiles_validation']
        if smiles_config['enabled']:
            min_len, max_len = smiles_config['min_length'], smiles_config['max_length']
            lengths = smiles_series.str.len()
            
            too_short = (lengths < min_len).sum()
            too_long = (lengths > max_len).sum()
            
            if too_short > 0:
                warnings.append(f"Found {too_short} SMILES shorter than {min_len} characters")
            if too_long > 0:
                warnings.append(f"Found {too_long} SMILES longer than {max_len} characters")
        
        return {'warnings': warnings, 'errors': errors}
    
    def _validate_target_column(self, target_series: pd.Series) -> Dict[str, List[str]]:
        """Validate target variable values."""
        warnings = []
        errors = []
        
        # Check for null values
        null_count = target_series.isnull().sum()
        if null_count > 0:
            errors.append(f"Found {null_count} null target values")
        
        # Check allowed values for classification
        allowed_values = self._validation_rules['target_validation']['allowed_values']
        if allowed_values is not None:
            unique_values = target_series.dropna().unique()
            invalid_values = [v for v in unique_values if v not in allowed_values]
            if invalid_values:
                errors.append(f"Invalid target values found: {invalid_values}")
        
        return {'warnings': warnings, 'errors': errors}
    
    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply preprocessing transformations to molecular data."""
        processed_data = data.copy()
        
        # Remove duplicates if enabled
        if self._validation_rules['duplicate_detection']['enabled']:
            initial_count = len(processed_data)
            processed_data = processed_data.drop_duplicates()
            removed_count = initial_count - len(processed_data)
            if removed_count > 0:
                self.logger.info(f"Removed {removed_count} duplicate records")
        
        # Filter out invalid SMILES if validation is enabled
        smiles_col = self.config.get('smiles_column', 'SMILES')
        if (smiles_col in processed_data.columns and 
            self._validation_rules['smiles_validation']['enabled']):
            
            initial_count = len(processed_data)
            processed_data = self._filter_invalid_smiles(processed_data, smiles_col)
            filtered_count = initial_count - len(processed_data)
            if filtered_count > 0:
                self.logger.info(f"Filtered out {filtered_count} records with invalid SMILES")
        
        # Normalize target values if needed
        target_col = self.config.get('target_column', 'TARGET')
        if target_col in processed_data.columns:
            processed_data = self._normalize_target_values(processed_data, target_col)
        
        return processed_data
    
    def _filter_invalid_smiles(self, data: pd.DataFrame, smiles_col: str) -> pd.DataFrame:
        """Filter out records with invalid SMILES strings."""
        # Basic filtering for null values and extreme lengths
        config = self._validation_rules['smiles_validation']
        
        valid_mask = (
            data[smiles_col].notna() &
            (data[smiles_col].str.len() >= config['min_length']) &
            (data[smiles_col].str.len() <= config['max_length'])
        )
        
        return data[valid_mask]
    
    def _normalize_target_values(self, data: pd.DataFrame, target_col: str) -> pd.DataFrame:
        """Normalize target values for consistency."""
        normalized_data = data.copy()
        
        # Convert string labels to numeric if needed
        if normalized_data[target_col].dtype == 'object':
            unique_values = normalized_data[target_col].unique()
            if set(unique_values) <= {'是', '否'}:
                normalized_data[target_col] = normalized_data[target_col].map({'是': 1, '否': 0})
                self.logger.info("Converted Chinese target labels to numeric values")
        
        return normalized_data