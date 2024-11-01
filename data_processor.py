import os
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
import re
import pickle
from typing import Dict, List, Optional, Tuple, Union

from config import (
    logger,
    CAT_THRESHOLD,
    MERGE_THRESHOLD,
    ENGINEERING_FEATURES,
    ORIGINAL_NUMERIC_FEATURES,
    ENGINEERED_FEATURES,
    CATEGORICAL_FEATURES
)

class DataLoader:
    """Handles data loading and initial preprocessing"""
    def __init__(self, file_path: str):
        self.file_path = file_path
        
    def load_data(self) -> pd.DataFrame:
        """Loads data from file and performs initial checks"""
        try:
            file_ext = os.path.splitext(self.file_path)[1].lower()
            
            if file_ext == '.csv':
                data = pd.read_csv(self.file_path)
            elif file_ext in ['.xlsx', '.xls']:
                data = pd.read_excel(self.file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_ext}")
                
            logger.info(f"Loaded {len(data)} rows and {len(data.columns)} columns")
            return data
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise

class ColumnProcessor:
    """Handles column normalization and validation"""
    @staticmethod
    def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
        """Normalizes column names by removing spaces and special characters"""
        df_copy = df.copy()
        df_copy.columns = [re.sub(r'[^a-zA-Z0-9]', '', col.strip()) for col in df_copy.columns]
        return df_copy

def find_closest_matches(search_column: str, available_columns: List[str]) -> List[str]:
    """Find closest matching column names"""
    from difflib import get_close_matches
    return get_close_matches(search_column, available_columns, n=3, cutoff=0.6)

def analyze_column_discrepancies(required_cols: List[str], available_cols: List[str]) -> None:
    """Analyze and report column name discrepancies"""
    logger.info("\nAnalyzing column name discrepancies:")
    for col in required_cols:
        if col not in available_cols:
            matches = find_closest_matches(col, available_cols)
            if matches:
                logger.info(f"Column '{col}' not found. Did you mean: {matches}?")
            else:
                logger.info(f"Column '{col}' not found and no close matches found.")


class FeaturePreprocessor(BaseEstimator, TransformerMixin):
    """Handles initial feature preprocessing including handling missing values and merging categories."""
    
    def __init__(
        self, 
        numeric_features: List[str], 
        categorical_features: List[str], 
        cat_threshold: float = CAT_THRESHOLD,
        merge_threshold: float = MERGE_THRESHOLD
    ):
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features
        self.cat_threshold = cat_threshold
        self.merge_threshold = merge_threshold
        self.num_imputer = SimpleImputer(strategy='median')
        self.cat_modes = {}
        self.category_maps = {}

    def fit(self, X, y=None):
        """Learn preprocessing parameters from training data."""
        data = X.copy()
        
        # Fit numeric imputer
        numeric_data = data[self.numeric_features].copy()
        self.num_imputer.fit(numeric_data)
        
        # Learn categorical handling parameters
        for col in self.categorical_features:
            if col in data.columns:
                missing_ratio = data[col].isnull().mean() * 100
                
                if missing_ratio > self.cat_threshold:
                    unique_cats = list(data[col].dropna().unique())
                    self.category_maps[col] = unique_cats + ['Unknown']
                else:
                    self.cat_modes[col] = data[col].mode()[0]
                    value_counts = data[col].value_counts(normalize=True) * 100
                    keep_categories = value_counts[value_counts >= self.merge_threshold].index.tolist()
                    
                    if any(value_counts < self.merge_threshold):
                        keep_categories.append('Other')
                    
                    self.category_maps[col] = keep_categories
                    
                    logger.info(f"\nColumn: {col}")
                    logger.info(f"Categories to keep: {keep_categories}")
                    logger.info(f"Number of categories after merging: {len(keep_categories)}")
        return self

    def transform(self, X):
        """Apply preprocessing to features."""
        data = X.copy()
        
        # Transform numeric features
        numeric_data = data[self.numeric_features].copy()
        data[self.numeric_features] = self.num_imputer.transform(numeric_data)
        
        # Transform categorical features
        for col in self.categorical_features:
            if col in data.columns:
                missing_ratio = data[col].isnull().mean() * 100
                
                if missing_ratio > self.cat_threshold:
                    data[col] = data[col].fillna('Unknown')
                else:
                    data[col] = data[col].fillna(self.cat_modes[col])
                    
                    known_categories = set(self.category_maps[col])
                    data[col] = data[col].apply(
                        lambda x: (x if x in known_categories else 'Other')
                        if 'Other' in known_categories
                        else (x if x in known_categories else next(iter(known_categories)))
                    )
        
        return data

class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Creates new features and removes original features used in engineering"""
    def __init__(self):
        self.original_features = None
        self.engineered_features = None
        self.features_to_drop = None
        
    def fit(self, X, y=None):
        """Identify features to be created and dropped"""
        X = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        
        self.original_features = X.columns.tolist()
        self.engineered_features = []
        self.features_to_drop = set()
        
        feature_combinations = [
            (['GrLivArea', 'TotalBsmtSF'], 'TotalSqFt'),
            (['YrSold', 'YearBuilt'], 'HouseAge'),
            (['FullBath', 'HalfBath', 'BsmtFullBath', 'BsmtHalfBath'], 'TotalBaths'),
            (['YrSold', 'YearRemodAdd'], 'YrRemodAge')
        ]
        
        for required_cols, new_feature in feature_combinations:
            if all(col in X.columns for col in required_cols):
                self.engineered_features.append(new_feature)
                self.features_to_drop.update(required_cols)
        
        return self

    def transform(self, X):
        """Apply feature engineering transformations"""
        X = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        data = X.copy()
        
        # Total Square Footage
        if all(col in data.columns for col in ['GrLivArea', 'TotalBsmtSF']):
            data['TotalSqFt'] = data['GrLivArea'].astype(float) + data['TotalBsmtSF'].fillna(0).astype(float)
        
        # House Age
        if all(col in data.columns for col in ['YrSold', 'YearBuilt']):
            data['HouseAge'] = data['YrSold'].astype(float) - data['YearBuilt'].astype(float)
        
        # Total Bathrooms
        bathroom_cols = ['FullBath', 'HalfBath', 'BsmtFullBath', 'BsmtHalfBath']
        if all(col in data.columns for col in bathroom_cols):
            data['TotalBaths'] = (
                data['FullBath'].astype(float) + 
                0.5 * data['HalfBath'].fillna(0).astype(float) + 
                data['BsmtFullBath'].fillna(0).astype(float) + 
                0.5 * data['BsmtHalfBath'].fillna(0).astype(float)
            )
        
        # Years since remodeling
        if all(col in data.columns for col in ['YrSold', 'YearRemodAdd']):
            data['YrRemodAge'] = data['YrSold'].astype(float) - data['YearRemodAdd'].astype(float)
        
        data = data.drop(columns=list(self.features_to_drop), errors='ignore')
        
        return data

class PreprocessingPipeline:
    """Handles creation and management of preprocessing pipeline"""
    def __init__(
        self,
        numeric_features: Optional[List[str]] = None,
        categorical_features: Optional[List[str]] = None,
        engineering_features: Optional[List[str]] = None
    ):
        self.numeric_features = numeric_features or ORIGINAL_NUMERIC_FEATURES
        self.categorical_features = categorical_features or CATEGORICAL_FEATURES
        self.engineering_features = engineering_features or ENGINEERING_FEATURES
        self.pipeline = None
        self.column_mapping = {}

    def _normalize_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize column names to match expected format"""
        df = df.copy()
        
        # Create mapping of original to normalized names
        normalized_columns = {}
        for col in df.columns:
            # Replace spaces and special characters with underscores
            normalized = re.sub(r'[^a-zA-Z0-9]', '_', col.strip())
            # Remove consecutive underscores
            normalized = re.sub(r'_+', '_', normalized)
            # Remove leading/trailing underscores
            normalized = normalized.strip('_')
            normalized_columns[col] = normalized
        
        # Store mapping for future reference
        self.column_mapping = normalized_columns
        
        # Rename columns
        df.rename(columns=normalized_columns, inplace=True)
        return df

    def _map_feature_names(self, features: List[str]) -> List[str]:
        """Map normalized feature names to original names"""
        # Create reverse mapping
        reverse_mapping = {v: k for k, v in self.column_mapping.items()}
        
        # Map features to original names
        mapped_features = []
        for feature in features:
            if feature in reverse_mapping:
                mapped_features.append(reverse_mapping[feature])
            else:
                # For engineered features that don't exist in original data
                mapped_features.append(feature)
        
        return mapped_features
    
    def _validate_columns(self, X: pd.DataFrame) -> None:
        """Validate that required columns exist in the dataset"""
        missing_cols = []
        all_required_cols = (
            self.engineering_features + 
            [f for f in self.numeric_features if f not in ENGINEERED_FEATURES] +
            self.categorical_features
        )

    def create_pipeline(self) -> Pipeline:
        """Creates the preprocessing pipeline"""
        numeric_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(
                drop='first',
                sparse_output=False,
                handle_unknown='ignore'
            ))
        ])
        
        initial_numeric_features = list(set(
            self.engineering_features + 
            [f for f in self.numeric_features if f not in ENGINEERED_FEATURES]
        ))
        
        initial_preprocessor = FeaturePreprocessor(
            numeric_features=self._map_feature_names(initial_numeric_features),
            categorical_features=self._map_feature_names(self.categorical_features)
        )
        
        final_preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, self._map_feature_names(self.numeric_features)),
                ('cat', categorical_transformer, self._map_feature_names(self.categorical_features))
            ],
            remainder='drop'
        )
        
        self.pipeline = Pipeline([
            ('initial_preprocessing', initial_preprocessor),
            ('feature_engineering', FeatureEngineer()),
            ('final_preprocessing', final_preprocessor)
        ])
        
        return self.pipeline
    
    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> np.ndarray:
        """Fit and transform the data using the pipeline"""
        # Normalize column names first
        X = self._normalize_column_names(X)
        
        if self.pipeline is None:
            self.create_pipeline()
        
        return self.pipeline.fit_transform(X, y)