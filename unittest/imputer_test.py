import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
import unittest

# MissingValueImputer class definition
class MissingValueImputer(BaseEstimator, TransformerMixin):
    def __init__(self, cat_threshold=0.5):
        self.cat_threshold = cat_threshold
        self.num_imputer = SimpleImputer(strategy = 'median')

    def fit(self, X):
        return self

    def transform(self, X):
        data = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)

        for col in data.select_dtypes(include=['object', 'category']).columns:
            missing_ratio = data[col].isnull().mean()
            if missing_ratio > self.cat_threshold:
                data[col] = data[col].astype('category').cat.add_categories(['Unknown']).fillna('Unknown')
            else:
                data[col] = data[col].fillna(data[col].mode()[0])

        num_cols = data.select_dtypes(include=['number']).columns
        data[num_cols] = self.num_imputer.fit_transform(data[num_cols])
        return data

# Unit test for MissingValueImputer
class TestMissingValueImputer(unittest.TestCase):
    def setUp(self):
        # Input data with missing values in both categorical and numerical columns
        self.input_data = pd.DataFrame({
            'col1': [1, 2, 3, 4, 5, 6],
            'col2': pd.Series(['A', 'A', 'A', 'B', 'A', np.nan]),  # Categorical column with missing value
            'col3': [1.0, 2.5, np.nan, 3.3, np.nan, 5.5]  # Numerical column with missing values
        })

        # Expected output after imputation
        self.expected_data = pd.DataFrame({
            'col1': [1, 2, 3, 4, 5, 6],
            'col2': pd.Series(['A', 'A', 'A', 'B', 'A', 'Unknown'], dtype = 'category'),  # Imputed 'Unknown' for missing value
            'col3': [1.0, 2.5, 2.9, 3.3, 2.9, 5.5]  # Imputed median for missing values
        })

        # Enforce consistent types across dataframes
        self.input_data = self.input_data.astype({'col1': 'float64', 'col3': 'float64'})
        self.expected_data = self.expected_data.astype({'col1': 'float64', 'col3': 'float64'})

        # Instantiate the imputer
        self.imputer = MissingValueImputer(cat_threshold = 0.1)

    def test_transform(self):
        # Perform the transformation
        transformed_data = self.imputer.transform(self.input_data)

        # Option 1: Explicitly convert types (already done in setUp)
        transformed_data = transformed_data.astype({'col1': 'float64', 'col3': 'float64'})

        # Check that the transformed dataframe matches the expected dataframe
        pd.testing.assert_frame_equal(transformed_data, self.expected_data, check_dtype=False)

# Running the test
if __name__ == '__main__':
    unittest.main()
