"""
Utility functions and helper classes for the UC Drug Activity Prediction system.

This module provides various utility functions for data manipulation, validation,
file operations, and system resource management that are used throughout
the application.
"""

import os
import sys
import json
import hashlib
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Callable
from contextlib import contextmanager
import time
import functools

class FileSystemManager:
    """
    Utility class for safe file system operations with validation and cleanup.
    """
    
    @staticmethod
    def ensure_directory(path: Union[str, Path], permissions: int = 0o755) -> Path:
        """
        Ensure directory exists with proper permissions.
        
        Args:
            path: Directory path to create
            permissions: Directory permissions (default: 755)
            
        Returns:
            Path object of created/existing directory
        """
        path_obj = Path(path)
        path_obj.mkdir(parents=True, exist_ok=True)
        
        if hasattr(os, 'chmod'):
            os.chmod(str(path_obj), permissions)
        
        return path_obj
    
    @staticmethod
    def safe_file_copy(src: Union[str, Path], dst: Union[str, Path], 
                      backup: bool = True) -> bool:
        """
        Safely copy file with optional backup of destination.
        
        Args:
            src: Source file path
            dst: Destination file path
            backup: Whether to backup existing destination file
            
        Returns:
            True if copy successful, False otherwise
        """
        src_path, dst_path = Path(src), Path(dst)
        
        if not src_path.exists():
            return False
        
        try:
            # Backup existing file if requested
            if backup and dst_path.exists():
                backup_path = dst_path.with_suffix(dst_path.suffix + '.bak')
                shutil.copy2(str(dst_path), str(backup_path))
            
            # Ensure destination directory exists
            FileSystemManager.ensure_directory(dst_path.parent)
            
            # Copy file
            shutil.copy2(str(src_path), str(dst_path))
            return True
            
        except Exception:
            return False
    
    @contextmanager
    def temporary_directory(self, prefix: str = "tmp_drug_activity_"):
        """
        Context manager for temporary directory creation and cleanup.
        
        Args:
            prefix: Prefix for temporary directory name
            
        Yields:
            Path object of temporary directory
        """
        temp_dir = None
        try:
            temp_dir = Path(tempfile.mkdtemp(prefix=prefix))
            yield temp_dir
        finally:
            if temp_dir and temp_dir.exists():
                shutil.rmtree(str(temp_dir), ignore_errors=True)

class DataValidationUtils:
    """
    Collection of data validation and sanitization utilities.
    """
    
    @staticmethod
    def validate_smiles_basic(smiles: str) -> Dict[str, Any]:
        """
        Perform basic SMILES string validation.
        
        Args:
            smiles: SMILES string to validate
            
        Returns:
            Dictionary with validation results
        """
        if not isinstance(smiles, str):
            return {'valid': False, 'error': 'SMILES must be string'}
        
        if not smiles.strip():
            return {'valid': False, 'error': 'Empty SMILES string'}
        
        # Basic character validation
        allowed_chars = set('CNOPSFClBrI[]()=+@1234567890-#/')
        invalid_chars = set(smiles) - allowed_chars
        
        if invalid_chars:
            return {
                'valid': False, 
                'error': f'Invalid characters: {invalid_chars}',
                'warnings': []
            }
        
        # Length validation
        if len(smiles) > 1000:
            return {
                'valid': True,
                'warnings': ['SMILES unusually long (>1000 chars)']
            }
        
        return {'valid': True, 'warnings': []}
    
    @staticmethod
    def sanitize_filename(filename: str, replacement_char: str = '_') -> str:
        """
        Sanitize filename by replacing invalid characters.
        
        Args:
            filename: Original filename
            replacement_char: Character to replace invalid chars with
            
        Returns:
            Sanitized filename
        """
        invalid_chars = '<>:"/\\|?*'
        sanitized = filename
        
        for char in invalid_chars:
            sanitized = sanitized.replace(char, replacement_char)
        
        # Remove multiple consecutive replacement chars
        while replacement_char * 2 in sanitized:
            sanitized = sanitized.replace(replacement_char * 2, replacement_char)
        
        return sanitized.strip(replacement_char)
    
    @staticmethod
    def calculate_data_fingerprint(data: Union[str, bytes, Path]) -> str:
        """
        Calculate SHA-256 fingerprint of data for integrity checking.
        
        Args:
            data: Data to fingerprint (string, bytes, or file path)
            
        Returns:
            Hexadecimal SHA-256 hash
        """
        hasher = hashlib.sha256()
        
        if isinstance(data, Path) or (isinstance(data, str) and Path(data).exists()):
            # File fingerprint
            with open(data, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hasher.update(chunk)
        elif isinstance(data, str):
            hasher.update(data.encode('utf-8'))
        elif isinstance(data, bytes):
            hasher.update(data)
        else:
            raise ValueError("Unsupported data type for fingerprinting")
        
        return hasher.hexdigest()

class PerformanceUtils:
    """
    Performance monitoring and optimization utilities.
    """
    
    @staticmethod
    def timing_decorator(func: Callable) -> Callable:
        """
        Decorator to measure function execution time.
        
        Args:
            func: Function to time
            
        Returns:
            Wrapped function with timing
        """
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                execution_time = time.perf_counter() - start_time
                print(f"{func.__name__} executed in {execution_time:.4f} seconds")
        
        return wrapper
    
    @staticmethod
    def memory_usage_monitor() -> Dict[str, float]:
        """
        Get current memory usage statistics.
        
        Returns:
            Dictionary with memory usage info
        """
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            
            return {
                'rss_mb': memory_info.rss / (1024 * 1024),
                'vms_mb': memory_info.vms / (1024 * 1024),
                'percent': process.memory_percent(),
                'available_gb': psutil.virtual_memory().available / (1024 * 1024 * 1024)
            }
        except ImportError:
            return {'error': 'psutil not available'}
    
    @contextmanager
    def resource_monitor(self, operation_name: str):
        """
        Context manager for monitoring resource usage during operations.
        
        Args:
            operation_name: Name of operation being monitored
        """
        start_memory = self.memory_usage_monitor()
        start_time = time.perf_counter()
        
        try:
            yield
        finally:
            end_time = time.perf_counter()
            end_memory = self.memory_usage_monitor()
            
            print(f"\\n{operation_name} Resource Usage:")
            print(f"  Execution time: {end_time - start_time:.4f}s")
            
            if 'error' not in start_memory and 'error' not in end_memory:
                memory_diff = end_memory['rss_mb'] - start_memory['rss_mb']
                print(f"  Memory change: {memory_diff:+.2f} MB")
                print(f"  Peak memory: {max(start_memory['rss_mb'], end_memory['rss_mb']):.2f} MB")

class ConfigurationUtils:
    """
    Utility functions for configuration management and validation.
    """
    
    @staticmethod
    def merge_configurations(*configs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Deep merge multiple configuration dictionaries.
        
        Args:
            *configs: Configuration dictionaries to merge
            
        Returns:
            Merged configuration dictionary
        """
        def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
            result = base.copy()
            
            for key, value in override.items():
                if (key in result and 
                    isinstance(result[key], dict) and 
                    isinstance(value, dict)):
                    result[key] = deep_merge(result[key], value)
                else:
                    result[key] = value
            
            return result
        
        result = {}
        for config in configs:
            result = deep_merge(result, config)
        
        return result
    
    @staticmethod
    def validate_configuration_schema(config: Dict[str, Any], 
                                    schema: Dict[str, Any]) -> List[str]:
        """
        Validate configuration against a simple schema.
        
        Args:
            config: Configuration to validate
            schema: Schema definition
            
        Returns:
            List of validation errors
        """
        errors = []
        
        def validate_recursive(cfg: Dict[str, Any], sch: Dict[str, Any], path: str = ""):
            for key, expected_type in sch.items():
                current_path = f"{path}.{key}" if path else key
                
                if key not in cfg:
                    errors.append(f"Missing required key: {current_path}")
                    continue
                
                if isinstance(expected_type, dict):
                    if not isinstance(cfg[key], dict):
                        errors.append(f"Expected dict at {current_path}, got {type(cfg[key])}")
                    else:
                        validate_recursive(cfg[key], expected_type, current_path)
                elif not isinstance(cfg[key], expected_type):
                    errors.append(f"Expected {expected_type.__name__} at {current_path}, got {type(cfg[key])}")
        
        validate_recursive(config, schema)
        return errors

class SystemUtils:
    """
    System-level utility functions.
    """
    
    @staticmethod
    def get_system_info() -> Dict[str, Any]:
        """
        Get comprehensive system information.
        
        Returns:
            Dictionary with system information
        """
        import platform
        
        info = {
            'platform': platform.system(),
            'platform_release': platform.release(),
            'platform_version': platform.version(),
            'architecture': platform.machine(),
            'processor': platform.processor(),
            'python_version': platform.python_version(),
            'python_implementation': platform.python_implementation(),
        }
        
        try:
            import psutil
            info.update({
                'cpu_count': psutil.cpu_count(),
                'memory_total_gb': psutil.virtual_memory().total / (1024**3),
                'disk_total_gb': psutil.disk_usage('/').total / (1024**3)
            })
        except ImportError:
            pass
        
        return info
    
    @staticmethod
    def check_dependencies() -> Dict[str, bool]:
        """
        Check availability of required dependencies.
        
        Returns:
            Dictionary mapping dependency names to availability
        """
        dependencies = [
            'pandas', 'numpy', 'sklearn', 'unimol_tools',
            'yaml', 'psutil'  # Optional dependencies
        ]
        
        availability = {}
        for dep in dependencies:
            try:
                __import__(dep)
                availability[dep] = True
            except ImportError:
                availability[dep] = False
        
        return availability

# Global utility instances for convenient access
fs_manager = FileSystemManager()
perf_utils = PerformanceUtils()
data_utils = DataValidationUtils()
config_utils = ConfigurationUtils()
system_utils = SystemUtils()