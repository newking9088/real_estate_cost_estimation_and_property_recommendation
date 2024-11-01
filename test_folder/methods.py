from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    learning_curve,
    train_test_split
)
from sklearn.metrics import mean_absolute_error, r2_score, mean_absolute_percentage_error
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import os
import time
import datetime
import logging
from typing import Dict, List, Tuple, Optional, Union
import seaborn as sns
from sklearn.base import clone

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
TRAIN_TEST_DELTA_THRESHOLD = 0.2
NUM_MODELS = 4
CV_FOLDS = 7

# Model name mapping and parameter grids
MODEL_NAME_MAPPING = {
    'rf': 'RandomForest',
    'lgbm': 'LightGBM',
    'cat': 'CatBoost',
    'ada': 'AdaBoost',
    'RandomForest': 'RandomForest',
    'LightGBM': 'LightGBM',
    'CatBoost': 'CatBoost',
    'AdaBoost': 'AdaBoost'
}

PARAM_GRIDS = {
    'RandomForest': {
        'max_depth': [5, 7, 9, 11, None],
        'n_estimators': [100, 200, 300],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    },
    'LightGBM': {
        'num_leaves': [31, 63, 127],
        'learning_rate': [0.01, 0.05, 0.1],
        'n_estimators': [100, 200, 300],
        'min_child_samples': [20, 50, 100],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0],
        'verbose': [-1]
    },
    'CatBoost': {
        'depth': [6, 8, 10],
        'learning_rate': [0.01, 0.05, 0.1],
        'iterations': [500, 1000],
        'l2_leaf_reg': [3, 5, 7],
        'subsample': [0.8, 0.9, 1.0]
    },
    'AdaBoost': {
        'n_estimators': [50, 100, 200, 300],
        'learning_rate': [0.01, 0.05, 0.1, 1.0],
        'loss': ['linear', 'square', 'exponential']
    }
}

class ModelTrainer:
    """Handles model training, evaluation and visualization"""
    def __init__(self):
        self.models = {
            'RandomForest': RandomForestRegressor(random_state=42),
            'LightGBM': LGBMRegressor(objective='regression_l1', random_state=42),
            'CatBoost': CatBoostRegressor(verbose=0, random_state=42),
            'AdaBoost': AdaBoostRegressor(random_state=42)
        }
        self.cv_results = {}

    def _perform_cross_validation(
        self,
        model,
        model_name: str,
        X: np.ndarray,
        y: np.ndarray
    ) -> Dict:
        """Perform k-fold cross-validation for a single model"""
        kf = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)
        cv_results = {
            'fold_scores': [],
            'train_scores': [],
            'val_scores': [],
            'train_mae': [],
            'val_mae': []
        }
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
            # Split data
            X_fold_train = X[train_idx]
            X_fold_val = X[val_idx]
            y_fold_train = y.iloc[train_idx]
            y_fold_val = y.iloc[val_idx]
            
            # Train model
            fold_model = clone(model)
            fold_model.fit(X_fold_train, y_fold_train)
            
            # Get predictions
            train_pred = fold_model.predict(X_fold_train)
            val_pred = fold_model.predict(X_fold_val)
            
            # Calculate metrics
            train_r2 = r2_score(y_fold_train, train_pred)
            val_r2 = r2_score(y_fold_val, val_pred)
            train_mae = mean_absolute_error(y_fold_train, train_pred)
            val_mae = mean_absolute_error(y_fold_val, val_pred)
            
            # Store results
            cv_results['fold_scores'].append({
                'fold': fold,
                'train_r2': train_r2,
                'val_r2': val_r2,
                'train_mae': train_mae,
                'val_mae': val_mae
            })
            cv_results['train_scores'].append(train_r2)
            cv_results['val_scores'].append(val_r2)
            cv_results['train_mae'].append(train_mae)
            cv_results['val_mae'].append(val_mae)
            
            logger.info(f"\nFold {fold} Results for {model_name}:")
            logger.info(f"Train R2: {train_r2:.4f}")
            logger.info(f"Validation R2: {val_r2:.4f}")
            logger.info(f"Train MAE: {train_mae:.4f}")
            logger.info(f"Validation MAE: {val_mae:.4f}")
        
        # Calculate averages
        cv_results['train_mean_r2'] = np.mean(cv_results['train_scores'])
        cv_results['train_std_r2'] = np.std(cv_results['train_scores'])
        cv_results['val_mean_r2'] = np.mean(cv_results['val_scores'])
        cv_results['val_std_r2'] = np.std(cv_results['val_scores'])
        cv_results['train_mean_mae'] = np.mean(cv_results['train_mae'])
        cv_results['val_mean_mae'] = np.mean(cv_results['val_mae'])
        
        return cv_results

    def plot_cv_results(self, cv_results: Dict[str, Dict]) -> None:
        """Plot cross-validation results for each model"""
        for model_name, results in cv_results.items():
            plt.figure(figsize=(15, 10))
            
            # Plot R² scores
            plt.subplot(2, 1, 1)
            plt.plot(results['train_scores'], 'o-', label='Train R²')
            plt.plot(results['val_scores'], 'o-', label='Validation R²')
            plt.axhline(y=results['train_mean_r2'], color='b', linestyle='--', alpha=0.5)
            plt.axhline(y=results['val_mean_r2'], color='orange', linestyle='--', alpha=0.5)
            plt.title(f'{model_name} Cross-Validation Scores')
            plt.xlabel('Fold')
            plt.ylabel('R² Score')
            plt.legend()
            plt.grid(True)
            
            # Plot MAE scores
            plt.subplot(2, 1, 2)
            plt.plot(results['train_mae'], 'o-', label='Train MAE')
            plt.plot(results['val_mae'], 'o-', label='Validation MAE')
            plt.axhline(y=results['train_mean_mae'], color='b', linestyle='--', alpha=0.5)
            plt.axhline(y=results['val_mean_mae'], color='orange', linestyle='--', alpha=0.5)
            plt.xlabel('Fold')
            plt.ylabel('MAE')
            plt.legend()
            plt.grid(True)
            
            plt.tight_layout()
            plt.show()

    def train_and_evaluate(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> pd.DataFrame:
        """Trains multiple models, performs CV, and evaluates performance"""
        results = []
        
        # Perform cross-validation for each model
        for name, model in self.models.items():
            logger.info(f"\nPerforming {CV_FOLDS}-fold CV for {name}")
            cv_result = self._perform_cross_validation(model, name, X_train, y_train)
            self.cv_results[name] = cv_result
            
            # Train final model
            start_time = time.time()
            model.fit(X_train, y_train)
            train_preds = model.predict(X_train)
            test_preds = model.predict(X_test)
            
            results.append({
                'Model': name,
                'Train MAE': mean_absolute_error(y_train, train_preds),
                'Test MAE': mean_absolute_error(y_test, test_preds),
                'Train R2': r2_score(y_train, train_preds),
                'Test R2': r2_score(y_test, test_preds),
                'Train MAPE': mean_absolute_percentage_error(y_train, train_preds),
                'Test MAPE': mean_absolute_percentage_error(y_test, test_preds),
                'CV Mean R2': cv_result['val_mean_r2'],
                'CV Std R2': cv_result['val_std_r2'],
                'Training Time': f"{time.time() - start_time:.2f}s"
            })
            
            # Plot CV results
            self.plot_cv_results({name: cv_result})
            
        return pd.DataFrame(results)
class ModelHypertuner:
    """Handles model hyperparameter tuning and validation analysis"""
    def __init__(
        self,
        models: Dict,
        param_grids: Optional[Dict] = None,
        train_test_delta_threshold: float = TRAIN_TEST_DELTA_THRESHOLD,
        n_best_models: Optional[Union[int, str, List[str]]] = NUM_MODELS,
        cv_folds: int = CV_FOLDS,
        scoring: str = 'r2',
        n_jobs: int = -1
    ):
        self.models = models
        self.param_grids = param_grids or PARAM_GRIDS
        self.delta_threshold = train_test_delta_threshold
        self.models_to_select = n_best_models
        self.cv_folds = cv_folds
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.best_models = {}
        self.cv_results = {}
        self.tuned_models = {}

    def _get_model_name(self, name: str) -> str:
        """Convert potential shorthand model names to full names"""
        return MODEL_NAME_MAPPING.get(name, name)

    def _filter_models(self) -> Dict:
        """Filter models based on models_to_select parameter"""
        if self.models_to_select is None:
            return self.models
            
        if isinstance(self.models_to_select, int):
            return self.models
            
        if isinstance(self.models_to_select, str):
            model_name = self._get_model_name(self.models_to_select)
            if model_name not in self.models:
                raise ValueError(f"Model {model_name} not found in available models")
            return {model_name: self.models[model_name]}
            
        if isinstance(self.models_to_select, (list, tuple)):
            selected_models = {}
            for name in self.models_to_select:
                model_name = self._get_model_name(name)
                if model_name not in self.models:
                    raise ValueError(f"Model {model_name} not found in available models")
                selected_models[model_name] = self.models[model_name]
            return selected_models
            
        raise ValueError("Invalid models_to_select parameter")

    def select_best_models(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Dict:
        """Select best models based on performance metrics"""
        available_models = self._filter_models()
        model_scores = []

        for name, model in available_models.items():
            model.fit(X_train, y_train)
            
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            train_r2 = r2_score(y_train, y_train_pred)
            test_r2 = r2_score(y_test, y_test_pred)
            score_delta = abs(train_r2 - test_r2)

            model_scores.append({
                'name': name,
                'model': model,
                'train_r2': train_r2,
                'test_r2': test_r2,
                'score_delta': score_delta
            })

        # Select models based on criteria
        valid_models = [
            model for model in model_scores 
            if model['score_delta'] <= self.delta_threshold
        ]

        if not valid_models:
            logger.warning(
                f"No models found with train-test delta below {self.delta_threshold}. "
                "Using all models."
            )
            valid_models = model_scores

        valid_models.sort(key=lambda x: x['test_r2'], reverse=True)
        n_models = self.models_to_select if isinstance(self.models_to_select, int) else 2
        selected_models = valid_models[:n_models]
        
        self.best_models = {
            model['name']: model['model'] for model in selected_models
        }

        return self.best_models

    def tune_hyperparameters(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> Dict:
        """Perform hyperparameter tuning using GridSearchCV"""
        logger.info("Starting hyperparameter tuning...")
        
        for name, model in self.best_models.items():
            if name not in self.param_grids:
                logger.warning(f"No parameter grid defined for {name}. Skipping tuning.")
                continue
                
            logger.info(f"Tuning {name}...")
            grid_search = GridSearchCV(
                estimator=model,
                param_grid=self.param_grids[name],
                cv=self.cv_folds,
                scoring=self.scoring,
                n_jobs=self.n_jobs,
                verbose=0
            )
            
            grid_search.fit(X, y)
            
            self.tuned_models[name] = grid_search.best_estimator_
            self.cv_results[name] = pd.DataFrame(grid_search.cv_results_)
            
            logger.info(
                f"Best parameters for {name}: {grid_search.best_params_}\n"
                f"Best score: {grid_search.best_score_:.4f}"
            )
        
        return self.tuned_models

    def plot_learning_curves(
        self,
        X: np.ndarray,
        y: np.ndarray,
        train_sizes: np.ndarray = np.linspace(0.1, 1.0, 5)
    ) -> None:
        """Plot learning curves for each selected model"""
        plt.figure(figsize=(15, 6 * len(self.best_models)))
        
        for idx, (name, model) in enumerate(self.best_models.items(), 1):
            train_sizes_abs, train_scores, test_scores = learning_curve(
                estimator=model,
                X=X,
                y=y,
                train_sizes=train_sizes,
                cv=self.cv_folds,
                n_jobs=self.n_jobs,
                scoring=self.scoring
            )
            
            train_mean = np.mean(train_scores, axis=1)
            train_std = np.std(train_scores, axis=1)
            test_mean = np.mean(test_scores, axis=1)
            test_std = np.std(test_scores, axis=1)
            
            plt.subplot(len(self.best_models), 1, idx)
            self._plot_learning_curve(
                train_sizes_abs, train_mean, train_std,
                test_mean, test_std, name
            )
        
        plt.tight_layout()
        plt.show()

    def _plot_learning_curve(
        self,
        train_sizes: np.ndarray,
        train_mean: np.ndarray,
        train_std: np.ndarray,
        test_mean: np.ndarray,
        test_std: np.ndarray,
        model_name: str
    ) -> None:
        """Helper function to plot a single learning curve"""
        plt.plot(train_sizes, train_mean, label='Training score')
        plt.plot(train_sizes, test_mean, label='Cross-validation score')
        
        plt.fill_between(
            train_sizes,
            train_mean - train_std,
            train_mean + train_std,
            alpha=0.1
        )
        plt.fill_between(
            train_sizes,
            test_mean - test_std,
            test_mean + test_std,
            alpha=0.1
        )
        
        plt.title(f'Learning Curves - {model_name}')
        plt.xlabel('Training Examples')
        plt.ylabel(f'Score ({self.scoring})')
        plt.legend(loc='best')
        plt.grid(True)

def save_models(base_models: Dict, tuned_models: Dict, model_dir: str = './models') -> None:
    """Save base and hypertuned models to specified directory"""
    os.makedirs(model_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    
    for name, model in base_models.items():
        filename = f"{name}_base_{timestamp}.pkl"
        path = os.path.join(model_dir, filename)
        with open(path, 'wb') as f:
            pickle.dump(model, f)
        logger.info(f"Saved base model: {filename}")
    
    for name, model in tuned_models.items():
        filename = f"{name}_tuned_{timestamp}.pkl"
        path = os.path.join(model_dir, filename)
        with open(path, 'wb') as f:
            pickle.dump(model, f)
        logger.info(f"Saved tuned model: {filename}")

def list_saved_models(model_dir: str = './models') -> pd.DataFrame:
    """List all saved models with their details"""
    if not os.path.exists(model_dir):
        return pd.DataFrame()
    
    models_info = []
    for filename in os.listdir(model_dir):
        if filename.endswith('.pkl'):
            name_parts = filename.replace('.pkl', '').split('_')
            model_name = name_parts[0]
            model_type = name_parts[1]
            timestamp = '_'.join(name_parts[2:])
            
            models_info.append({
                'Model Name': model_name,
                'Type': model_type,
                'Timestamp': timestamp,
                'Filename': filename,
                'Path': os.path.join(model_dir, filename)
            })
    
    return pd.DataFrame(models_info)


def plot_feature_importance(importance_df: pd.DataFrame, model_name: str) -> None:
    """Plot feature importance for a given model"""
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