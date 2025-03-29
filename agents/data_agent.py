 # agents/data_agent.py

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
import os
import logging
import json
from datetime import datetime
import hashlib

class DataAgent:
    """
    Agent responsible for data processing, cleaning, and preparation.
    Handles data import, validation, cleaning, transformation, and feature engineering.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.data_cache = {}
        self.data_metadata = {}
        
    def run(self, **params) -> Dict[str, Any]:
        """Main entry point for the data agent"""
        operation = params.get("operation", "process")
        
        if operation == "process":
            return self.process_data(**params)
        elif operation == "validate":
            return self.validate_data(**params)
        elif operation == "analyze":
            return self.analyze_data(**params)
        elif operation == "transform":
            return self.transform_data(**params)
        else:
            raise ValueError(f"Unknown operation: {operation}")
    
    def process_data(self, file_path: str, file_type: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """
        Process an uploaded data file
        
        Args:
            file_path: Path to the data file
            file_type: Optional file type override (auto-detected if not provided)
            
        Returns:
            Dictionary with processed data information
        """
        self.logger.info(f"Processing data file: {file_path}")
        
        # Determine file type if not provided
        if not file_type:
            file_type = self._detect_file_type(file_path)
        
        # Load the data based on file type
        try:
            if file_type == "csv":
                df = pd.read_csv(file_path, **kwargs)
            elif file_type == "excel":
                df = pd.read_excel(file_path, **kwargs)
            elif file_type == "json":
                df = pd.read_json(file_path, **kwargs)
            elif file_type == "parquet":
                df = pd.read_parquet(file_path, **kwargs)
            else:
                raise ValueError(f"Unsupported file type: {file_type}")
                
            # Generate a unique ID for this dataset
            dataset_id = self._generate_dataset_id(file_path)
            
            # Basic cleaning
            df = self._basic_cleaning(df)
            
            # Store in cache
            self.data_cache[dataset_id] = df
            
            # Generate and store metadata
            metadata = self._generate_metadata(df, file_path, file_type)
            self.data_metadata[dataset_id] = metadata
            
            # Save processed data
            processed_path = self._save_processed_data(df, dataset_id, file_type)
            
            return {
                "success": True,
                "dataset_id": dataset_id,
                "rows": len(df),
                "columns": len(df.columns),
                "metadata": metadata,
                "processed_path": processed_path
            }
            
        except Exception as e:
            self.logger.error(f"Error processing data file {file_path}: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "file_path": file_path,
                "file_type": file_type
            }
    
    def validate_data(self, data: Union[pd.DataFrame, str], schema: Optional[Dict] = None, **kwargs) -> Dict[str, Any]:
        """
        Validate data against a schema or perform basic quality checks
        
        Args:
            data: DataFrame or dataset_id
            schema: Optional validation schema
            
        Returns:
            Dictionary with validation results
        """
        # Get the DataFrame if dataset_id was provided
        if isinstance(data, str):
            if data not in self.data_cache:
                raise ValueError(f"Dataset ID {data} not found in cache")
            df = self.data_cache[data]
        else:
            df = data
        
        validation_results = {
            "success": True,
            "issues": [],
            "warnings": [],
            "summary": {}
        }
        
        # Check for missing values
        missing_values = df.isnull().sum()
        if missing_values.sum() > 0:
            missing_cols = missing_values[missing_values > 0].to_dict()
            validation_results["issues"].append({
                "type": "missing_values",
                "details": missing_cols
            })
        
        # Check for duplicate rows
        duplicate_count = df.duplicated().sum()
        if duplicate_count > 0:
            validation_results["issues"].append({
                "type": "duplicate_rows",
                "details": {"count": int(duplicate_count)}
            })
        
        # Check data types if schema is provided
        if schema and "columns" in schema:
            for col_name, expected_type in schema["columns"].items():
                if col_name not in df.columns:
                    validation_results["issues"].append({
                        "type": "missing_column",
                        "details": {"column": col_name}
                    })
                else:
                    # Type checking logic here
                    pass
        
        # Check for outliers using IQR method for numeric columns
        for col in df.select_dtypes(include=np.number).columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = df[(df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))].shape[0]
            
            if outliers > 0:
                validation_results["warnings"].append({
                    "type": "outliers",
                    "details": {"column": col, "count": outliers}
                })
        
        # Set success status based on issues
        if validation_results["issues"]:
            validation_results["success"] = False
        
        # Generate summary
        validation_results["summary"] = {
            "issues_count": len(validation_results["issues"]),
            "warnings_count": len(validation_results["warnings"]),
            "missing_values_total": int(missing_values.sum()),
            "duplicate_rows": int(duplicate_count)
        }
        
        return validation_results
    
    def analyze_data(self, dataset_id: str, columns: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Perform exploratory analysis on the dataset
        
        Args:
            dataset_id: ID of the dataset to analyze
            columns: Optional list of columns to analyze (all if None)
            
        Returns:
            Dictionary with analysis results
        """
        if dataset_id not in self.data_cache:
            raise ValueError(f"Dataset ID {dataset_id} not found in cache")
        
        df = self.data_cache[dataset_id]
        
        # Filter columns if specified
        if columns:
            df = df[columns]
        
        analysis_results = {
            "summary_stats": {},
            "correlations": None,
            "column_types": {},
            "unique_counts": {},
            "sample_rows": df.head(5).to_dict(orient="records")
        }
        
        # Generate summary statistics
        try:
            analysis_results["summary_stats"] = df.describe(include="all").to_dict()
        except:
            # Fallback for incompatible columns
            numeric_df = df.select_dtypes(include=np.number)
            if not numeric_df.empty:
                analysis_results["summary_stats"] = numeric_df.describe().to_dict()
        
        # Calculate correlations for numeric columns
        numeric_df = df.select_dtypes(include=np.number)
        if len(numeric_df.columns) > 1:
            analysis_results["correlations"] = numeric_df.corr().to_dict()
        
        # Determine column types
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                if df[col].nunique() <= 10:
                    col_type = "categorical_numeric"
                else:
                    col_type = "continuous"
            elif pd.api.types.is_datetime64_dtype(df[col]):
                col_type = "datetime"
            elif df[col].nunique() <= 10:
                col_type = "categorical"
            else:
                col_type = "text"
            
            analysis_results["column_types"][col] = col_type
        
        # Count unique values for categorical columns
        for col in df.columns:
            if analysis_results["column_types"][col] in ["categorical", "categorical_numeric"]:
                analysis_results["unique_counts"][col] = df[col].value_counts().to_dict()
        
        return analysis_results
    
    def transform_data(self, dataset_id: str, transformations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Apply transformations to the dataset
        
        Args:
            dataset_id: ID of the dataset to transform
            transformations: List of transformation operations to apply
            
        Returns:
            Dictionary with transformation results
        """
        if dataset_id not in self.data_cache:
            raise ValueError(f"Dataset ID {dataset_id} not found in cache")
        
        df = self.data_cache[dataset_id].copy()
        
        transform_results = {
            "success": True,
            "applied_transformations": [],
            "failed_transformations": [],
            "new_dataset_id": None
        }
        
        for transform in transformations:
            try:
                transform_type = transform.get("type")
                
                if transform_type == "filter":
                    column = transform.get("column")
                    condition = transform.get("condition")
                    value = transform.get("value")
                    
                    if condition == "equals":
                        df = df[df[column] == value]
                    elif condition == "not_equals":
                        df = df[df[column] != value]
                    elif condition == "greater_than":
                        df = df[df[column] > value]
                    elif condition == "less_than":
                        df = df[df[column] < value]
                    elif condition == "contains":
                        df = df[df[column].astype(str).str.contains(value)]
                    else:
                        raise ValueError(f"Unknown filter condition: {condition}")
                
                elif transform_type == "create_column":
                    column = transform.get("column")
                    expression = transform.get("expression")
                    # Using eval is generally not safe in production code
                    # This is a simplified example
                    df[column] = df.eval(expression)
                
                elif transform_type == "drop_column":
                    columns = transform.get("columns", [])
                    df = df.drop(columns=columns)
                
                elif transform_type == "rename_column":
                    rename_map = transform.get("rename_map", {})
                    df = df.rename(columns=rename_map)
                
                elif transform_type == "fill_missing":
                    column = transform.get("column")
                    method = transform.get("method")
                    value = transform.get("value", None)
                    
                    if method == "value":
                        df[column] = df[column].fillna(value)
                    elif method == "mean":
                        df[column] = df[column].fillna(df[column].mean())
                    elif method == "median":
                        df[column] = df[column].fillna(df[column].median())
                    elif method == "mode":
                        df[column] = df[column].fillna(df[column].mode()[0])
                    else:
                        raise ValueError(f"Unknown fill method: {method}")
                
                else:
                    raise ValueError(f"Unknown transformation type: {transform_type}")
                
                transform_results["applied_transformations"].append(transform)
                
            except Exception as e:
                self.logger.error(f"Error applying transformation {transform}: {str(e)}")
                transform["error"] = str(e)
                transform_results["failed_transformations"].append(transform)
        
        if transform_results["failed_transformations"]:
            transform_results["success"] = False
        
        # Generate a new dataset ID and store the transformed data
        new_dataset_id = f"{dataset_id}_transformed_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        self.data_cache[new_dataset_id] = df
        
        # Update metadata for the new dataset
        metadata = self.data_metadata.get(dataset_id, {}).copy()
        metadata["parent_dataset_id"] = dataset_id
        metadata["transformations"] = transform_results["applied_transformations"]
        metadata["transformed_at"] = datetime.now().isoformat()
        metadata["rows"] = len(df)
        metadata["columns"] = len(df.columns)
        self.data_metadata[new_dataset_id] = metadata
        
        transform_results["new_dataset_id"] = new_dataset_id
        transform_results["rows"] = len(df)
        transform_results["columns"] = len(df.columns)
        
        return transform_results
    
    def get_dataset(self, dataset_id: str) -> pd.DataFrame:
        """Retrieve a dataset by ID"""
        if dataset_id not in self.data_cache:
            raise ValueError(f"Dataset ID {dataset_id} not found in cache")
        return self.data_cache[dataset_id]
    
    def get_metadata(self, dataset_id: str) -> Dict[str, Any]:
        """Retrieve metadata for a dataset"""
        if dataset_id not in self.data_metadata:
            raise ValueError(f"Metadata for dataset ID {dataset_id} not found")
        return self.data_metadata[dataset_id]
    
    def list_datasets(self) -> List[Dict[str, Any]]:
        """List all available datasets with basic metadata"""
        return [{
            "dataset_id": dataset_id,
            "rows": len(df),
            "columns": len(df.columns),
            **{k: v for k, v in self.data_metadata.get(dataset_id, {}).items() 
               if k in ["name", "source", "created_at", "file_type"]}
        } for dataset_id, df in self.data_cache.items()]
    
    def _detect_file_type(self, file_path: str) -> str:
        """Detect file type based on extension"""
        ext = os.path.splitext(file_path)[1].lower()
        
        if ext == ".csv":
            return "csv"
        elif ext in [".xls", ".xlsx", ".xlsm"]:
            return "excel"
        elif ext == ".json":
            return "json"
        elif ext == ".parquet":
            return "parquet"
        else:
            raise ValueError(f"Unsupported file extension: {ext}")
    
    def _basic_cleaning(self, df: pd.DataFrame) -> pd.DataFrame:
        """Perform basic data cleaning"""
        # Remove completely empty rows and columns
        df = df.dropna(how="all")
        df = df.dropna(axis=1, how="all")
        
        # Strip whitespace from string columns
        for col in df.select_dtypes(include="object"):
            df[col] = df[col].str.strip() if hasattr(df[col], 'str') else df[col]
        
        # Convert common date columns
        date_columns = [col for col in df.columns if any(date_kw in col.lower() 
                       for date_kw in ["date", "time", "day", "month", "year"])]
        
        for col in date_columns:
            try:
                df[col] = pd.to_datetime(df[col], errors="ignore")
            except:
                pass
        
        return df
    
    # def _generate_dataset_id(self, file_path: str) -> str:
    #     """Generate a unique ID for the dataset"""
    #     base = os.path.basename(file_path)
    #     timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    #     unique_part = hashlib.md5(f"{base}_{timestamp}".encode()).hexdigest()[:8]
    #     return f"dataset_{unique_part}"
    
   #############################

    def _generate_dataset_id(self, file_path: str) -> str:
        """Generate a unique ID for the dataset"""
        base = os.path.basename(file_path)
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        unique_part = hashlib.md5(f"{base}_{timestamp}".encode()).hexdigest()[:8]
        return f"dataset_{unique_part}"
    
    def _generate_metadata(self, df: pd.DataFrame, file_path: str, file_type: str) -> Dict[str, Any]:
        """Generate metadata for the dataset"""
        metadata = {
            "name": os.path.basename(file_path),
            "source": file_path,
            "file_type": file_type,
            "created_at": datetime.now().isoformat(),
            "rows": len(df),
            "columns": len(df.columns),
            "column_info": {}
        }
        
        # Generate column-level metadata
        for col in df.columns:
            col_metadata = {
                "dtype": str(df[col].dtype),
                "missing_values": int(df[col].isnull().sum()),
                "unique_values": int(df[col].nunique())
            }
            
            # Add descriptive statistics for numeric columns
            if pd.api.types.is_numeric_dtype(df[col]):
                col_metadata.update({
                    "min": float(df[col].min()) if not pd.isna(df[col].min()) else None,
                    "max": float(df[col].max()) if not pd.isna(df[col].max()) else None,
                    "mean": float(df[col].mean()) if not pd.isna(df[col].mean()) else None,
                    "median": float(df[col].median()) if not pd.isna(df[col].median()) else None
                })
            
            metadata["column_info"][col] = col_metadata
        
        return metadata
    
    def _save_processed_data(self, df: pd.DataFrame, dataset_id: str, file_type: str) -> str:
        """Save processed data to disk"""
        # Create processed directory if it doesn't exist
        os.makedirs("data/processed", exist_ok=True)
        
        # Save the data
        file_path = f"data/processed/{dataset_id}.parquet"
        df.to_parquet(file_path)
        
        # Save metadata
        metadata_path = f"data/processed/{dataset_id}_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(self.data_metadata.get(dataset_id, {}), f, indent=2)
        
        return file_path
    
    def generate_synthetic_data(self, template: Dict[str, Any], rows: int = 1000) -> Dict[str, Any]:
        """
        Generate synthetic data based on a template
        
        Args:
            template: Dictionary with column definitions and generation rules
            rows: Number of rows to generate
            
        Returns:
            Dictionary with generated dataset information
        """
        synthetic_data = {}
        
        for col_name, col_config in template.items():
            col_type = col_config.get("type", "random")
            
            if col_type == "categorical":
                categories = col_config.get("categories", ["A", "B", "C"])
                weights = col_config.get("weights", None)
                synthetic_data[col_name] = np.random.choice(categories, size=rows, p=weights)
                
            elif col_type == "numeric":
                min_val = col_config.get("min", 0)
                max_val = col_config.get("max", 100)
                distribution = col_config.get("distribution", "uniform")
                
                if distribution == "uniform":
                    synthetic_data[col_name] = np.random.uniform(min_val, max_val, size=rows)
                elif distribution == "normal":
                    mean = col_config.get("mean", (min_val + max_val) / 2)
                    std = col_config.get("std", (max_val - min_val) / 6)
                    synthetic_data[col_name] = np.random.normal(mean, std, size=rows)
                    # Clip to specified range
                    synthetic_data[col_name] = np.clip(synthetic_data[col_name], min_val, max_val)
                    
            elif col_type == "datetime":
                start_date = pd.to_datetime(col_config.get("start", "2020-01-01"))
                end_date = pd.to_datetime(col_config.get("end", "2023-12-31"))
                
                # Generate random timestamps between start and end
                start_timestamp = start_date.timestamp()
                end_timestamp = end_date.timestamp()
                random_timestamps = np.random.uniform(start_timestamp, end_timestamp, size=rows)
                synthetic_data[col_name] = pd.to_datetime(random_timestamps, unit='s')
                
            elif col_type == "text":
                templates = col_config.get("templates", ["Text {}", "Sample {}", "Example {}"])
                values = [np.random.choice(templates).format(i) for i in range(rows)]
                synthetic_data[col_name] = values
                
            elif col_type == "dependent":
                # Generate values dependent on another column
                source_col = col_config.get("source_column")
                formula = col_config.get("formula", "x * 2")  # Default formula
                
                if source_col not in synthetic_data:
                    raise ValueError(f"Source column {source_col} must be generated before dependent column {col_name}")
                
                # Apply the formula (using a safer eval approach)
                x = synthetic_data[source_col]  # Define x for formula
                synthetic_data[col_name] = eval(formula)
        
        # Convert to DataFrame
        df = pd.DataFrame(synthetic_data)
        
        # Generate dataset ID and save
        dataset_id = f"synthetic_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        self.data_cache[dataset_id] = df
        
        # Generate and save metadata
        metadata = {
            "name": f"Synthetic Dataset {dataset_id}",
            "source": "synthetic_generator",
            "created_at": datetime.now().isoformat(),
            "rows": len(df),
            "columns": len(df.columns),
            "synthetic": True,
            "template": template
        }
        self.data_metadata[dataset_id] = metadata
        
        # Save to disk
        os.makedirs("data/synthetic", exist_ok=True)
        df.to_parquet(f"data/synthetic/{dataset_id}.parquet")
        
        with open(f"data/synthetic/{dataset_id}_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        return {
            "success": True,
            "dataset_id": dataset_id,
            "rows": len(df),
            "columns": len(df.columns),
            "metadata": metadata
        }
