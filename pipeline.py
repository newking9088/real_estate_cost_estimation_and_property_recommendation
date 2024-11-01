import pandas as pd
import numpy as np
import logging
import os
from typing import Dict, List, Tuple
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

class DataLoader:
    """Handles data loading and initial preprocessing"""
    def __init__(self, file_path: str):
        self.file_path = file_path
        
    def load_data(self) -> pd.DataFrame:
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
    @staticmethod
    def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
        df_copy = df.copy()
        df_copy.columns = [re.sub(r'[^a-zA-Z0-9]', '', col.strip()) for col in df_copy.columns]
        return df_copy

class FeaturePreprocessor(BaseEstimator, TransformerMixin):
    def __init__(
        self, 
        numeric_features: List[str], 
        categorical_features: List[str],
        cat_threshold: float = 10,
        merge_threshold: float = 8
    ):
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features
        self.cat_threshold = cat_threshold
        self.merge_threshold = merge_threshold
        self.num_imputer = SimpleImputer(strategy='median')
        self.cat_modes = {col: 'Unknown' for col in categorical_features}  # Initialize with default
        self.category_maps = {col: ['Unknown'] for col in categorical_features}  # Initialize with default

    def fit(self, X, y=None):
        data = X.copy()
        
        # Handle missing columns
        for col in self.categorical_features:
            if col not in data.columns:
                data[col] = 'Unknown'
                logger.info(f"Added missing column {col} with 'Unknown' values")
        
        # Fit numeric imputer
        numeric_data = data[self.numeric_features].copy()
        self.num_imputer.fit(numeric_data)
        
        # Learn categorical parameters
        for col in self.categorical_features:
            if col in data.columns:
                data[col] = data[col].fillna('Unknown')  # Fill NAs before computing stats
                value_counts = data[col].value_counts(normalize=True) * 100
                
                # Keep categories above threshold
                keep_categories = value_counts[value_counts >= self.merge_threshold].index.tolist()
                
                # Add 'Other' if there are categories below threshold
                if any(value_counts < self.merge_threshold):
                    keep_categories.append('Other')
                    
                # Add 'Unknown' to all category maps
                if 'Unknown' not in keep_categories:
                    keep_categories.append('Unknown')
                
                self.category_maps[col] = keep_categories
                self.cat_modes[col] = data[col].mode().iloc[0] if not data[col].empty else 'Unknown'
                
                logger.info(f"\nColumn: {col}")
                logger.info(f"Categories to keep: {keep_categories}")
                logger.info(f"Mode value: {self.cat_modes[col]}")
                
        return self

    def transform(self, X):
        data = X.copy()
        
        # Handle missing columns
        for col in self.categorical_features:
            if col not in data.columns:
                data[col] = 'Unknown'
        
        # Transform numeric features
        numeric_data = data[self.numeric_features].copy()
        data[self.numeric_features] = self.num_imputer.transform(numeric_data)
        
        # Transform categorical features
        for col in self.categorical_features:
            # Fill missing values
            data[col] = data[col].fillna('Unknown')
            
            # Map rare categories to 'Other'
            known_categories = set(self.category_maps[col])
            data[col] = data[col].apply(lambda x: x if x in known_categories else 'Other')
            
            # Ensure all values are in the known categories
            invalid_categories = set(data[col].unique()) - known_categories
            if invalid_categories:
                data[col] = data[col].replace(invalid_categories, 'Other')
        
        return data

class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Creates new features and removes original features used in engineering"""
    def __init__(self):
        self.original_features = None
        self.engineered_features = None
        self.features_to_drop = None
        
    def fit(self, X, y=None):
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
        
        logger.info("\n=== Feature Engineering Information ===")
        logger.info(f"Original features to be dropped: {list(self.features_to_drop)}")
        logger.info(f"New engineered features: {self.engineered_features}")
        
        return self

    def transform(self, X):
        X = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        data = X.copy()
        
        # Create engineered features
        if all(col in data.columns for col in ['GrLivArea', 'TotalBsmtSF']):
            data['TotalSqFt'] = data['GrLivArea'].astype(float) + data['TotalBsmtSF'].fillna(0).astype(float)
        
        if all(col in data.columns for col in ['YrSold', 'YearBuilt']):
            data['HouseAge'] = data['YrSold'].astype(float) - data['YearBuilt'].astype(float)
        
        bath_cols = ['FullBath', 'HalfBath', 'BsmtFullBath', 'BsmtHalfBath']
        if all(col in data.columns for col in bath_cols):
            data['TotalBaths'] = (
                data['FullBath'].astype(float) + 
                0.5 * data['HalfBath'].fillna(0).astype(float) + 
                data['BsmtFullBath'].fillna(0).astype(float) + 
                0.5 * data['BsmtHalfBath'].fillna(0).astype(float)
            )
        
        if all(col in data.columns for col in ['YrSold', 'YearRemodAdd']):
            data['YrRemodAge'] = data['YrSold'].astype(float) - data['YearRemodAdd'].astype(float)
        
        # Drop original features
        data = data.drop(columns=list(self.features_to_drop), errors='ignore')
        
        return data

class PreprocessingPipeline:
    def __init__(
        self, 
        numeric_features: List[str], 
        categorical_features: List[str],
        engineering_features: List[str] = None
    ):
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features
        self.engineering_features = engineering_features or []
        self.pipeline = None
        self.transformed_data = None
        
    def create_pipeline(self) -> Pipeline:
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
        
        initial_preprocessor = FeaturePreprocessor(
            numeric_features=self.numeric_features + self.engineering_features,
            categorical_features=self.categorical_features
        )
        
        feature_engineer = FeatureEngineer()
        
        final_preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, self.numeric_features),
                ('cat', categorical_transformer, self.categorical_features)
            ],
            remainder='passthrough'  # Changed from 'drop' to 'passthrough'
        )
        
        return Pipeline([
            ('initial_preprocessing', initial_preprocessor),
            ('feature_engineering', feature_engineer),
            ('final_preprocessing', final_preprocessor)
        ])
    
    def fit_transform(self, data: pd.DataFrame) -> np.ndarray:
        try:
            logger.info("Starting preprocessing pipeline...")
            data = ColumnProcessor.normalize_column_names(data)
            self.pipeline = self.create_pipeline()
            transformed_data = self.pipeline.fit_transform(data)
            
            logger.info("Preprocessing completed successfully")
            self._log_feature_info()
            
            return transformed_data
            
        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            raise
    
    def transform(self, data: pd.DataFrame) -> np.ndarray:
        if self.pipeline is None:
            raise ValueError("Pipeline has not been fitted. Call fit_transform first.")
            
        try:
            data = ColumnProcessor.normalize_column_names(data)
            return self.pipeline.transform(data)
        except Exception as e:
            logger.error(f"Transform failed: {str(e)}")
            raise
            
    def get_feature_names(self) -> Dict[str, List[str]]:
        if self.pipeline is None:
            raise ValueError("Pipeline has not been fitted. Call fit_transform first.")
            
        try:
            numeric_features = self.pipeline.named_steps['final_preprocessing'].named_transformers_['num'].get_feature_names_out()
            categorical_features = self.pipeline.named_steps['final_preprocessing'].named_transformers_['cat'].get_feature_names_out()
            feature_engineer = self.pipeline.named_steps['feature_engineering']
            engineered_features = feature_engineer.engineered_features if hasattr(feature_engineer, 'engineered_features') else []
            
            return {
                'numeric_features': list(numeric_features),
                'categorical_features': list(categorical_features),
                'engineered_features': engineered_features,
                'all_features': list(numeric_features) + list(categorical_features)
            }
        except Exception as e:
            logger.error(f"Error getting feature names: {str(e)}")
            raise
            
    def _log_feature_info(self):
        feature_info = self.get_feature_names()
        logger.info("\nFeature Information:")
        logger.info(f"Original numeric features: {len(self.numeric_features)}")
        logger.info(f"Engineered features: {len(feature_info['engineered_features'])}")
        logger.info(f"Categorical features (encoded): {len(feature_info['categorical_features'])}")
        logger.info(f"Total features: {len(feature_info['all_features'])}")