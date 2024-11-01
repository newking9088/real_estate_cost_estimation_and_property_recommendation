import os
import argparse
import warnings
import numpy as np
from sklearn.model_selection import train_test_split

from config import logger, MODELS
from data_processor import DataLoader, PreprocessingPipeline
from model_trainer import ModelSelector, ModelTuner, ModelPersistence, ModelVisualizer

def validate_paths(args):
    """Validate input and output paths"""
    if args.data_path and not os.path.exists(args.data_path):
        raise FileNotFoundError(f"Data file not found: {args.data_path}")
    
    if args.load_path and not os.path.exists(args.load_path):
        raise FileNotFoundError(f"Load path not found: {args.load_path}")
    
    if args.save_path:
        os.makedirs(args.save_path, exist_ok=True)

def validate_model_names(model_names):
    """Validate provided model names"""
    if not model_names:
        return
    
    invalid_models = [name for name in model_names if name not in MODELS]
    if invalid_models:
        raise ValueError(
            f"Invalid model names: {invalid_models}. "
            f"Available models: {list(MODELS.keys())}"
        )

def parse_args():
    parser = argparse.ArgumentParser(description='House Price Prediction Pipeline')
    parser.add_argument('--data_path', type=str, required=True,
                      help='Path to the input data file')
    parser.add_argument('--mode', type=str, required=True,
                      choices=['preprocess', 'train', 'tune', 'predict'],
                      help='Pipeline mode to run')
    parser.add_argument('--model_names', type=str, nargs='+',
                      help='Names of models to use (e.g., RandomForest LightGBM)')
    parser.add_argument('--save_path', type=str,
                      help='Path to save processed data or models')
    parser.add_argument('--load_path', type=str,
                      help='Path to load saved pipeline or models')
    parser.add_argument('--cv', action='store_true',
                      help='Use cross-validation during training')
    return parser.parse_args()

def preprocess_data(args):
    """Run data preprocessing pipeline"""
    try:
        # Load data
        loader = DataLoader(args.data_path)
        data = loader.load_data()
        
        # Print column names for debugging
        logger.info("Available columns in dataset:")
        for col in data.columns:
            logger.info(f"  - {col}")
        
        # Create and run preprocessing pipeline
        preprocessor = PreprocessingPipeline()
        X = data.drop('SalePrice', axis=1)
        y = data['SalePrice']
        
        # Fit and transform the data
        X_processed = preprocessor.fit_transform(X)
        
        # Save the pipeline and processed data if path provided
        if args.save_path:
            os.makedirs(args.save_path, exist_ok=True)
            preprocessor.save_pipeline(os.path.join(args.save_path, 'preprocessing_pipeline.pkl'))
            np.save(os.path.join(args.save_path, 'X_processed.npy'), X_processed)
            np.save(os.path.join(args.save_path, 'y.npy'), y)
            logger.info(f"Saved preprocessing pipeline and data to {args.save_path}")
        
        return X_processed, y, preprocessor
        
    except Exception as e:
        logger.error(f"Error in preprocessing: {str(e)}")
        raise

def train_models(args, X=None, y=None):
    """Train selected models"""
    try:
        # Load processed data if not provided
        if X is None or y is None:
            if not args.load_path:
                raise ValueError("Must provide either data or load_path")
            X = np.load(os.path.join(args.load_path, 'X_processed.npy'))
            y = np.load(os.path.join(args.load_path, 'y.npy'))
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Initialize model selector with specified models
        selector = ModelSelector(model_names=args.model_names)
        
        # Train models
        results = selector.train(X_train, y_train, cv=args.cv)
        
        # Save trained models if path provided
        if args.save_path:
            os.makedirs(args.save_path, exist_ok=True)
            for name, result in results.items():
                ModelPersistence.save_model(
                    result['model'],
                    os.path.join(args.save_path, f'{name}_base.pkl'),
                    metadata={'cv_results': result.get('cv_results')}
                )
            logger.info(f"Saved trained models to {args.save_path}")
        
        return results, (X_train, X_test, y_train, y_test)
        
    except Exception as e:
        logger.error(f"Error in model training: {str(e)}")
        raise


def predict(args):
    """Make predictions using saved models"""
    if not args.load_path:
        raise ValueError("Must provide load_path for prediction")
    
    # Load preprocessing pipeline
    preprocessor = PreprocessingPipeline.load_pipeline(
        os.path.join(args.load_path, 'preprocessing_pipeline.pkl')
    )
    
    # Load and preprocess new data
    loader = DataLoader(args.data_path)
    new_data = loader.load_data()
    X_new = preprocessor.transform(new_data.drop('SalePrice', axis=1))
    
    # Load models and make predictions
    predictions = {}
    for model_name in args.model_names:
        # Try to load tuned model first, fall back to base model
        try:
            model_path = os.path.join(args.load_path, f'{model_name}_tuned.pkl')
            if not os.path.exists(model_path):
                model_path = os.path.join(args.load_path, f'{model_name}_base.pkl')
            model, metadata = ModelPersistence.load_model(model_path)
            predictions[model_name] = model.predict(X_new)
        except Exception as e:
            logger.error(f"Error loading or predicting with {model_name}: {e}")
    
    return predictions

def tune_models(args, X=None, y=None):
    """Tune hyperparameters for selected models"""
    # Load processed data if not provided
    if X is None or y is None:
        if not args.load_path:
            raise ValueError("Must provide either data or load_path")
        X = np.load(os.path.join(args.load_path, 'X_processed.npy'))
        y = np.load(os.path.join(args.load_path, 'y.npy'))
    
    # Split data if needed
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Load base models if provided
    base_models = {}
    if args.load_path:
        for model_name in args.model_names:
            model_path = os.path.join(args.load_path, f'{model_name}_base.pkl')
            if os.path.exists(model_path):
                model, _ = ModelPersistence.load_model(model_path)
                base_models[model_name] = model
    
    # Initialize tuner
    tuner = ModelTuner()
    
    # Tune each model
    tuned_results = {}
    for name in args.model_names:
        model = base_models.get(name, MODELS[name])
        tuned_results[name] = tuner.tune_model(model, name, X_train, y_train)
    
    # Save tuned models if path provided
    if args.save_path:
        os.makedirs(args.save_path, exist_ok=True)
        for name, result in tuned_results.items():
            ModelPersistence.save_model(
                result['model'],
                os.path.join(args.save_path, f'{name}_tuned.pkl'),
                metadata={
                    'best_params': result['best_params'],
                    'cv_results': result['cv_results']
                }
            )
        logger.info(f"Saved tuned models to {args.save_path}")
    
    return tuned_results

def main():
    """Main function to run the pipeline"""
    try:
        # Parse and validate arguments
        args = parse_args()
        validate_paths(args)
        validate_model_names(args.model_names)
        
        # Suppress warnings
        warnings.filterwarnings('ignore')
        
        # Execute pipeline based on mode
        if args.mode == 'preprocess':
            X_processed, y, preprocessor = preprocess_data(args)
            logger.info("Preprocessing completed successfully")
            
        elif args.mode == 'train':
            if args.load_path:
                results, data_split = train_models(args)
            else:
                X_processed, y, preprocessor = preprocess_data(args)
                results, data_split = train_models(args, X_processed, y)
            logger.info("Training completed successfully")
            
        elif args.mode == 'tune':
            if args.load_path:
                results = tune_models(args)
            else:
                X_processed, y, preprocessor = preprocess_data(args)
                results = tune_models(args, X_processed, y)
            logger.info("Tuning completed successfully")
            
        elif args.mode == 'predict':
            predictions = predict(args)
            logger.info("Predictions completed successfully")
            if args.save_path:
                prediction_path = os.path.join(args.save_path, 'predictions.npy')
                np.save(prediction_path, predictions)
                logger.info(f"Saved predictions to {prediction_path}")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()