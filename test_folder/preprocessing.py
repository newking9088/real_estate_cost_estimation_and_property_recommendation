import os
import pandas as pd
import numpy as np
import logging
from typing import Dict, List
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
CAT_THRESHOLD = 10
MERGE_THRESHOLD = 8

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

    def get_feature_names(self, pipeline) -> Dict[str, List[str]]:
        """Get feature names after preprocessing and one-hot encoding"""
        try:
            categorical_encoded = []
            for col in self.categorical_features:
                if col in self.category_maps:
                    categories = self.category_maps[col][1:]
                    encoded_names = [f"{col}_{cat}" for cat in categories]
                    categorical_encoded.extend(encoded_names)

            feature_engineer = pipeline.named_steps['feature_engineering']
            engineered_features = feature_engineer.engineered_features if hasattr(feature_engineer, 'engineered_features') else []
            
            numeric_features = [f for f in self.numeric_features if f not in feature_engineer.features_to_drop]
            numeric_features.extend(engineered_features)
            
            feature_info = {
                'numeric_features': numeric_features,
                'categorical_features': categorical_encoded,
                'total_features': len(numeric_features) + len(categorical_encoded)
            }
            
            return feature_info
            
        except Exception as e:
            logger.error(f"Error getting feature names: {str(e)}")
            raise

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