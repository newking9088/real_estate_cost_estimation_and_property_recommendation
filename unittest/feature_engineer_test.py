import unittest
import pandas as pd
import warnings

class FeatureEngineer:
    def __init__(self, data_type_dict):
        self.data_type_dict = data_type_dict

    def create_features(self, data):
        used_features = ['GrLivArea', 'TotalBsmtSF', 'YrSold', 'YearRemodAdd', 
                         'YearBuilt', 'GarageYrBlt', 'FullBath', 
                         'HalfBath', 'BsmtFullBath', 'BsmtHalfBath']
        
        # Check if all used features are in data columns
        missing_features = [feature for feature in used_features if feature not in data.columns]
        if missing_features:
            raise KeyError(f"Missing columns: {', '.join(missing_features)}")
        
        # Create new features only if the required columns are present
        if 'GrLivArea' in data.columns and 'TotalBsmtSF' in data.columns:
            data['TotalSqFt'] = data['GrLivArea'].fillna(0) + data['TotalBsmtSF'].fillna(0)
        else:
            missing_cols = []
            if 'GrLivArea' not in data.columns:
                missing_cols.append('GrLivArea')
            if 'TotalBsmtSF' not in data.columns:
                missing_cols.append('TotalBsmtSF')
            raise KeyError(f"Columns not found: {', '.join(missing_cols)}")
        
        if 'YrSold' in data.columns and 'YearRemodAdd' in data.columns:
            data['YrRemodAge'] = data['YrSold'].fillna(0) - data['YearRemodAdd'].fillna(0)
        else:
            missing_cols = []
            if 'YrSold' not in data.columns:
                missing_cols.append('YrSold')
            if 'YearRemodAdd' not in data.columns:
                missing_cols.append('YearRemodAdd')
            raise KeyError(f"Columns not found: {', '.join(missing_cols)}")
        
        if 'YearBuilt' in data.columns:
            data['HouseAge'] = 2010 - data['YearBuilt'].fillna(0)
        else:
            raise KeyError("Column not found: YearBuilt")
        
        if 'GarageYrBlt' in data.columns:
            data['GarageAge'] = data['GarageYrBlt'].fillna(data['YearBuilt']).apply(lambda x: 2010 - x)
        else:
            raise KeyError("Column not found: GarageYrBlt")
        
        data['TotalBaths'] = (data['FullBath'].fillna(0) + 
                              0.5 * data['HalfBath'].fillna(0) + 
                              data['BsmtFullBath'].fillna(0) + 
                              0.5 * data['BsmtHalfBath'].fillna(0))

        # Get the desired features that exist in data
        desired_features = set(self.data_type_dict.keys()).intersection(data.columns) - set(used_features)
        return data[list(desired_features)]

class TestFeatureEngineer(unittest.TestCase):
    def setUp(self):
        # Sample data for testing
        self.data_type_dict = {
            'TotalSqFt': 'number',
            'GarageAge': 'number',
            'TotalBaths': 'number',
            'YrRemodAge': 'number',
            'HouseAge': 'number'
        }
        self.feature_engineer = FeatureEngineer(self.data_type_dict)
        self.test_data = pd.DataFrame({
            'GrLivArea': [1500, 2000, 1800],
            'TotalBsmtSF': [800, 600, None],
            'YrSold': [2008, 2007, 2009],
            'YearRemodAdd': [2000, 1999, 2005],
            'YearBuilt': [1990, 1995, 2000],
            'GarageYrBlt': [2005, 2002, None],
            'FullBath': [2, 2, 1],
            'HalfBath': [1, None, 1],
            'BsmtFullBath': [1, 1, None],
            'BsmtHalfBath': [0, 0, 1]
        })

    def assert_frame_equal_floats(self, df1, df2):
        # Convert all numeric columns to float
        df1 = df1.astype(float)
        df2 = df2.astype(float)
        pd.testing.assert_frame_equal(df1.sort_index(axis=1), df2.sort_index(axis=1))

    def test_feature_creation(self):
        transformed_data = self.feature_engineer.create_features(self.test_data)

        # Expected results
        expected_data = {
            'TotalSqFt': [2300.0, 2600.0, 1800.0],
            'YrRemodAge': [8, 8, 4],   # Can be int or float
            'HouseAge': [20, 15, 10],  # Can be int or float
            'GarageAge': [5, 8, 10],    # Can be int or float
            'TotalBaths': [3.5, 3, 2]
        }
        expected_df = pd.DataFrame(expected_data)

        # Use the custom assertion function to compare
        self.assert_frame_equal_floats(transformed_data, expected_df)

    def test_returned_features(self):
        transformed_data = self.feature_engineer.create_features(self.test_data)
        expected_features = set(self.data_type_dict.keys())

        # Verify that the transformed data contains the expected features
        self.assertTrue(expected_features.issubset(transformed_data.columns))

if __name__ == '__main__':
    unittest.main()
