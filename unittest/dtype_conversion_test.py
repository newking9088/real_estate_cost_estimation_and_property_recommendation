import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
import unittest

class DataValidator(BaseEstimator, TransformerMixin):
    def __init__(self, data_types, message=False):
        self.data_types = data_types
        self.message = message

    def fit(self, X):
        return self

    def transform(self, X):
        for column, expected_dtype in self.data_types.items():
            if column in X.columns:
                actual_dtype = X[column].dtype
                if expected_dtype == 'number' and not pd.api.types.is_numeric_dtype(actual_dtype):
                    if self.message:
                        print(f"Converting column '{column}' to numeric.")
                    X[column] = pd.to_numeric(X[column], errors='coerce').astype(float)  # Ensure float
                elif expected_dtype == 'category' and not isinstance(actual_dtype, pd.CategoricalDtype):
                    if self.message:
                        print(f"Converting column '{column}' to category.")
                    X[column] = X[column].astype('category')
        return X


# Unit test for DataValidator
class TestDataValidator(unittest.TestCase):
    def setUp(self):
        self.data = pd.DataFrame({
            'col1': ['1', '2', '3', '4'],  # Should be converted to numeric
            'col2': ['A', 'B', 'C', 'D'],  # Should be converted to category
            'col3': [10, 20, 30, 40]        # Already numeric
        })

        self.expected_data = pd.DataFrame({
            'col1': [1.0, 2.0, 3.0, 4.0],  # Converted to numeric (float)
            'col2': pd.Series(['A', 'B', 'C', 'D'], dtype='category'),  # Converted to category
            'col3': [10, 20, 30, 40]        # Remains numeric
        })

        self.data_types = {
            'col1': 'number',
            'col2': 'category',
            'col3': 'number'
        }

        self.validator = DataValidator(data_types=self.data_types)

    def test_transform(self):
        transformed_data = self.validator.transform(self.data)
        pd.testing.assert_frame_equal(transformed_data, self.expected_data)

    def test_missing_column(self):
        missing_column_data = self.data.drop(columns=['col2'])
        transformed_data = self.validator.transform(missing_column_data)
        # col1 should still be converted and col3 should remain the same
        expected_missing_data = pd.DataFrame({
            'col1': [1.0, 2.0, 3.0, 4.0],  # Converted to numeric
            'col3': [10, 20, 30, 40]        # Remains numeric
        })
        pd.testing.assert_frame_equal(transformed_data[['col1', 'col3']], expected_missing_data)

    def test_no_conversion_needed(self):
        valid_data = pd.DataFrame({
            'col1': [1, 2, 3, 4],  # Already numeric
            'col2': pd.Series(['A', 'B', 'C', 'D'], dtype='category'),  # Already category
            'col3': [10, 20, 30, 40]  # Already numeric
        })
        transformed_data = self.validator.transform(valid_data)
        pd.testing.assert_frame_equal(transformed_data, valid_data)

# Running the test
if __name__ == '__main__':
    unittest.main()
