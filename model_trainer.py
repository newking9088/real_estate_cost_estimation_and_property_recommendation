"""Model training and evaluation utilities"""
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.base import clone
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
from typing import Dict, List, Optional, Union

from config import (
    logger,
    MODELS,
    PARAM_GRIDS,
    CV_FOLDS,
    TRAIN_TEST_DELTA_THRESHOLD
)

class ModelSelector:
    """Handles model selection and training"""
    def __init__(
        self,
        models: Optional[Dict] = None,
        model_names: Optional[Union[str, List[str]]] = None
    ):
        self.available_models = models or MODELS
        self.selected_models = self._filter_models(model_names) if model_names else self.available_models
        
    def _filter_models(self, model_names: Union[str, List[str]]) -> Dict:
        """Filter models based on names"""
        if isinstance(model_names, str):
            model_names = [model_names]
        
        selected = {}
        for name in model_names:
            if name in self.available_models:
                selected[name] = self.available_models[name]
            else:
                logger.warning(f"Model {name} not found in available models")
        
        return selected
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        cv: bool = True
    ) -> Dict:
        """Train selected models"""
        results = {}
        for name, model in self.selected_models.items():
            if cv:
                results[name] = self._train_with_cv(model, name, X_train, y_train)
            else:
                model.fit(X_train, y_train)
                results[name] = {'model': model}
        return results
    
    def _train_with_cv(self, model, name: str, X: np.ndarray, y: np.ndarray) -> Dict:
        """Train model with cross-validation"""
        # Implementation of cross-validation training
        pass

class ModelTuner:
    """Handles model hyperparameter tuning"""
    def __init__(
        self,
        param_grids: Optional[Dict] = None,
        cv_folds: int = CV_FOLDS
    ):
        self.param_grids = param_grids or PARAM_GRIDS
        self.cv_folds = cv_folds
        self.tuned_models = {}
        
    def tune_model(
        self,
        model,
        model_name: str,
        X: np.ndarray,
        y: np.ndarray
    ) -> Dict:
        """Tune hyperparameters for a single model"""
        if model_name not in self.param_grids:
            logger.warning(f"No parameter grid for {model_name}. Using default parameters.")
            return {'model': model}
            
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=self.param_grids[model_name],
            cv=self.cv_folds,
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X, y)
        self.tuned_models[model_name] = grid_search.best_estimator_
        
        return {
            'model': grid_search.best_estimator_,
            'best_params': grid_search.best_params_,
            'cv_results': pd.DataFrame(grid_search.cv_results_)
        }

class ModelPersistence:
    """Handles model saving and loading"""
    @staticmethod
    def save_model(model, filepath: str, metadata: Optional[Dict] = None) -> None:
        """Save model with optional metadata"""
        data = {
            'model': model,
            'metadata': metadata or {}
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
    
    @staticmethod
    def load_model(filepath: str) -> tuple:
        """Load model and metadata"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        return data['model'], data['metadata']

class ModelVisualizer:
    """Handles model visualization and analysis"""
    @staticmethod
    def plot_feature_importance(importance_df: pd.DataFrame, model_name: str) -> None:
        """Plot feature importance"""
        plt.figure(figsize=(12, 6))
        sns.barplot(
            data=importance_df.head(10),
            x='Relative Importance (%)',
            y='Feature',
            palette='viridis'
        )
        plt.title(f'Top 10 Important Features - {model_name}')
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_cv_results(cv_results: Dict) -> None:
        """Plot cross-validation results"""
        # Implementation of CV results visualization
        pass
