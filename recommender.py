# recommender.py
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from pipeline import PreprocessingPipeline, DataLoader, logger

class HouseRecommender:
    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors
        self.pipeline = None
        self.knn_model = None
        self.data = None
        self.feature_names = None
        self.required_columns = None
        self.engineered_features = None
        self.display_features = None
        self.original_data = None
        self.transformed_data = None

    def setup_pipeline(self):
        """Define features and create pipeline"""
        # Features for engineering
        engineering_features = [
            'GrLivArea', 'TotalBsmtSF', 'YrSold', 'YearBuilt',
            'FullBath', 'HalfBath', 'BsmtFullBath', 'BsmtHalfBath',
            'YearRemodAdd'
        ]
        
        # Core numeric features
        numeric_features = [
            'OverallQual', 'TotRmsAbvGrd', 'GarageCars',
            'Fireplaces', 'LotFrontage', 'MasVnrArea',
            'LotArea', 'GarageArea', 'BsmtFinSF1'
        ]
        
        # Engineered features
        self.engineered_features = [
            'TotalSqFt', 'HouseAge', 'TotalBaths', 'YrRemodAge'
        ]
        
        # Essential categorical features
        categorical_features = [
            'Neighborhood', 'ExterQual', 'BsmtQual', 'KitchenQual', 'FireplaceQu',
            'GarageType', 'SaleType', 'SaleCondition'
        ]
        
        # Features to display in recommendations
        self.display_features = [
            'SalePrice', 'Neighborhood', 'GrLivArea', 'TotalBsmtSF',
            'OverallQual', 'YearBuilt', 'TotRmsAbvGrd'
        ]
        
        # Store required columns
        self.required_columns = set(numeric_features + categorical_features + engineering_features)
        
        # Create pipeline
        self.pipeline = PreprocessingPipeline(
            numeric_features=numeric_features,
            categorical_features=categorical_features,
            engineering_features=engineering_features
        )

    def fit(self, data_path):
        """Fit the pipeline and KNN model"""
        try:
            logger.info("Starting model fitting process...")
            
            # Load data
            data_loader = DataLoader(data_path)
            self.original_data = data_loader.load_data()
            self.data = self.original_data.copy()
            
            # Setup and fit pipeline
            self.setup_pipeline()
            
            # Transform data
            logger.info("Transforming data through pipeline...")
            self.transformed_data = self.pipeline.fit_transform(self.data)
            
            # Get feature names
            self.feature_names = self.pipeline.get_feature_names()
            
            # Print transformed data info
            logger.info(f"Transformed data shape: {self.transformed_data.shape}")
            
            # Fit KNN model
            logger.info("Fitting KNN model...")
            self.knn_model = NearestNeighbors(
                n_neighbors=self.n_neighbors,
                metric='euclidean'
            )
            self.knn_model.fit(self.transformed_data)
            
            logger.info("Model fitting completed successfully!")
            return self
            
        except Exception as e:
            logger.error(f"Error fitting model: {str(e)}")
            raise

    def get_recommendations(self, house_idx, return_distance=True):
        """Get similar houses for a given house index"""
        if self.pipeline is None or self.knn_model is None:
            raise ValueError("Model not fitted. Call fit() first.")
            
        try:
            # Get the house data
            house_data = self.original_data.iloc[[house_idx]].copy()
            
            # Transform the data
            house_transformed = self.pipeline.transform(house_data)
            
            # Find nearest neighbors
            distances, indices = self.knn_model.kneighbors(house_transformed)
            
            # Get recommendations from original data
            recommendations = self.original_data.iloc[indices[0]].copy()
            
            # Add distance if requested
            if return_distance:
                recommendations['Distance'] = distances[0]
            
            # Get available display features
            available_features = [col for col in self.display_features if col in recommendations.columns]
            if return_distance:
                available_features.append('Distance')
                
            return recommendations[available_features]
            
        except Exception as e:
            logger.error(f"Error getting recommendations: {str(e)}")
            raise

    def get_house_details(self, house_idx):
        """Get detailed information about a specific house"""
        if self.original_data is None:
            raise ValueError("Model not fitted. Call fit() first.")
            
        try:
            house = self.original_data.iloc[house_idx]
            
            details = {
                'Basic Information': {
                    'Sale Price': f"${house['SalePrice']:,.2f}",
                    'Neighborhood': house['Neighborhood'],
                    'Year Built': int(house['YearBuilt']),
                    'Overall Quality': int(house['OverallQual'])
                },
                'Size Information': {
                    'Total Living Area': f"{house['GrLivArea']:,.0f} sqft",
                    'Total Basement': f"{house['TotalBsmtSF']:,.0f} sqft",
                    'Total Rooms': int(house['TotRmsAbvGrd'])
                }
            }
            
            return details
            
        except Exception as e:
            logger.error(f"Error getting house details: {str(e)}")
            raise

def main():
    """Example usage of the recommender"""
    try:
        # Initialize and fit recommender
        recommender = HouseRecommender(n_neighbors=5)
        recommender.fit('./data/raw_data/AmesHousing.csv')
        
        # Get recommendations for a specific house
        house_idx = 0
        
        # Get and display house details
        details = recommender.get_house_details(house_idx)
        print("\nOriginal House Details:")
        for category, items in details.items():
            print(f"\n{category}:")
            for key, value in items.items():
                print(f"{key}: {value}")
        
        # Get and display recommendations
        similar_houses = recommender.get_recommendations(house_idx)
        print("\nSimilar Houses:")
        print(similar_houses)
        
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        raise

if __name__ == "__main__":
    main()