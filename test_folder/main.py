import argparse
import logging
import warnings
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from preprocessing import (
    DataLoader,
    ColumnProcessor,
    FeaturePreprocessor,
    FeatureEngineer
)
from methods import (
    ModelTrainer,
    ModelHypertuner,
    save_models,
    list_saved_models
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class HousePricePipeline:
    """Main pipeline class that orchestrates the entire process"""
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.data_loader = DataLoader(file_path)
        self.column_processor = ColumnProcessor()
        
        # Define feature groups
        self.engineering_features = [
            'GrLivArea', 'TotalBsmtSF',
            'YrSold', 'YearBuilt',
            'FullBath', 'HalfBath', 'BsmtFullBath', 'BsmtHalfBath',
            'YearRemodAdd'
        ]
        
        self.original_numeric_features = [
            'OverallQual',
            'TotRmsAbvGrd',
            'GarageCars',
            'Fireplaces',
            'LotFrontage',
            'MasVnrArea'
        ]
        
        self.engineered_features = [
            'TotalSqFt',
            'HouseAge',
            'TotalBaths',
            'YrRemodAge'
        ]
        
        self.numeric_features = self.original_numeric_features + self.engineered_features
        
        self.categorical_features = [
            'Neighborhood',
            'FireplaceQu',
            'KitchenQual',
            'BsmtExposure'
        ]

    def create_preprocessing_pipeline(self) -> Pipeline:
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
            [f for f in self.numeric_features if f not in self.engineered_features]
        ))
        
        initial_preprocessor = FeaturePreprocessor(
            numeric_features=initial_numeric_features,
            categorical_features=self.categorical_features
        )
        
        final_preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, self.numeric_features),
                ('cat', categorical_transformer, self.categorical_features)
            ],
            remainder='drop'
        )
        
        return Pipeline([
            ('initial_preprocessing', initial_preprocessor),
            ('feature_engineering', FeatureEngineer()),
            ('final_preprocessing', final_preprocessor)
        ])

    def run(self):
        """Runs the entire pipeline"""
        try:
            logger.info("Starting house price prediction pipeline...")
            
            # Load and preprocess data
            data = self.data_loader.load_data()
            data = self.column_processor.normalize_column_names(data)
            
            # Split features and target
            X = data.drop('SalePrice', axis=1)
            y = data['SalePrice']
            
            # Create train-test split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            # Create and fit preprocessing pipeline
            preprocessing_pipeline = self.create_preprocessing_pipeline()
            X_train_processed = preprocessing_pipeline.fit_transform(X_train)
            X_test_processed = preprocessing_pipeline.transform(X_test)
            
            # Initialize trainer and hypertuner
            trainer = ModelTrainer()
            hypertuner = ModelHypertuner(trainer.models)
            
            # Train and evaluate initial models
            initial_results = trainer.train_and_evaluate(
                X_train_processed, y_train,
                X_test_processed, y_test
            )
            logger.info("\nInitial Model Results:")
            print(initial_results)
            
            # Select best models and tune hyperparameters
            best_models = hypertuner.select_best_models(
                X_train_processed, y_train,
                X_test_processed, y_test
            )
            
            tuned_models = hypertuner.tune_hyperparameters(
                X_train_processed, y_train
            )
            
            # Save models
            save_models(trainer.models, tuned_models)
            
            # Display saved models
            saved_models = list_saved_models()
            if not saved_models.empty:
                logger.info("\nSaved Models:")
                print(saved_models)
            
            return {
                'pipeline': preprocessing_pipeline,
                'initial_results': initial_results,
                'best_models': best_models,
                'tuned_models': tuned_models
            }
            
        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            raise

def main():
    """Main function to run the pipeline"""
    parser = argparse.ArgumentParser(description='Run house price prediction pipeline')
    parser.add_argument(
        '--data_path',
        type=str,
        default='./data/raw_data/AmesHousing.csv',
        help='Path to the input data file'
    )
    args = parser.parse_args()
    
    # Suppress warnings
    warnings.filterwarnings('ignore')
    
    try:
        pipeline = HousePricePipeline(args.data_path)
        results = pipeline.run()
        logger.info("Pipeline completed successfully")
        return results
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()