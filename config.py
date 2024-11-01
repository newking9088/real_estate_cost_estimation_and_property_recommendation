import logging
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

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
CAT_THRESHOLD = 10
MERGE_THRESHOLD = 8

# Feature groups - updated to match actual column names in the dataset
# Feature groups - these should match exactly with your dataset's column names
ENGINEERING_FEATURES = [
    'Gr Liv Area', 'Total Bsmt SF',
    'Yr Sold', 'Year Built',
    'Full Bath', 'Half Bath', 'Bsmt Full Bath', 'Bsmt Half Bath',
    'Year Remod/Add'
]

ORIGINAL_NUMERIC_FEATURES = [
    'OverallQual',
    'TotRmsAbvGrd',
    'GarageCars',
    'Fireplaces',
    'LotFrontage',
    'MasVnrArea'
]

ENGINEERED_FEATURES = [
    'TotalSqFt',
    'HouseAge',
    'TotalBaths',
    'YrRemodAge'
]

CATEGORICAL_FEATURES = [
    'Neighborhood',
    'FireplaceQu',
    'KitchenQual',
    'BsmtExposure'
]
# Model Configurations
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


MODELS = {
    'RandomForest': RandomForestRegressor(random_state=42),
    'LightGBM': LGBMRegressor(objective='regression_l1', random_state=42),
    'CatBoost': CatBoostRegressor(verbose=0, random_state=42),
    'AdaBoost': AdaBoostRegressor(random_state=42)
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