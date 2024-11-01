import unittest
from unittest.mock import patch
import pandas as pd


def load_data(file_path):
    data = pd.read_csv(file_path)
    data.columns = data.columns.str.replace(' ', '').str.replace(r'[^A-Za-z0-9_]', '', regex=True)
    return data

desired_features = ['TotalSqFt',
 'HouseAge',
 'OverallQual',
 'TotalBaths',
 'TotRmsAbvGrd',
 'GarageCars',
 'YrRemodAge',
 'Fireplaces',
 'LotFrontage',
 'MasVnrArea',
 'Neighborhood',
 'FireplaceQu',
 'KitchenQual',
 'BsmtExposure']


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
            raise KeyError(f"Columns not found: {', '.join(missing_features)}")
        
        # Create new features only if the required columns are present
        if 'GrLivArea' in data.columns and 'TotalBsmtSF' in data.columns:
            data['TotalSqFt'] = data['GrLivArea'].fillna(0) + data['TotalBsmtSF'].fillna(0)
        else:
            raise KeyError(f"Columns not found: GrLivArea, TotalBsmtSF")
        
        if 'YrSold' in data.columns and 'YearRemodAdd' in data.columns:
            data['YrRemodAge'] = data['YrSold'].fillna(0) - data['YearRemodAdd'].fillna(0)
        else:
            raise KeyError(f"Columns not found: YrSold, YearRemodAdd")
        
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
        return data[[feature for feature in desired_features if feature in data.columns]]


class TestFeatureEngineer(unittest.TestCase):
    def setUp(self):
        # Sample data for testing
        self.sample_data = pd.DataFrame({
            'GrLivArea': [2000, 1500, None],
            'TotalBsmtSF': [800, 600, 400],
            'YrSold': [2010, 2010, 2010],
            'YearRemodAdd': [2000, 2005, None],
            'YearBuilt': [2000, 1995, 1990],
            'GarageYrBlt': [2000, None, 1980],
            'FullBath': [2, 1, 1],
            'HalfBath': [1, 0, 0],
            'BsmtFullBath': [1, 0, 0],
            'BsmtHalfBath': [0, 0, 1]
        })

        # Initialize data_type_dict
        self.data_type_dict = {
            'GrLivArea': 'number',
            'YearBuilt': 'number',
            'YrSold': 'number',
            'GarageYrBlt': 'number',
            'TotalBsmtSF': 'number',
            'TotalSqFt': 'number',
            'HouseAge': 'number',
            'OverallQual': 'number',
            'TotalBaths': 'number',
            'TotRmsAbvGrd': 'number',
            'GarageCars': 'number',
            'YrRemodAge': 'number',
            'Fireplaces': 'number',
            'LotFrontage': 'number',
            'MasVnrArea': 'number',
            'Neighborhood': 'category',
            'FireplaceQu': 'category',
            'KitchenQual': 'category',
            'BsmtExposure': 'category',
            'SalePrice': 'number'
        }

        # Initialize FeatureEngineer
        self.feature_engineer = FeatureEngineer(self.data_type_dict)

    def test_create_features_success(self):
        result = self.feature_engineer.create_features(self.sample_data.copy())
        self.assertIn('TotalSqFt', result.columns)
        self.assertIn('HouseAge', result.columns)
        self.assertEqual(result['TotalSqFt'][0], 2800.0)

    def test_create_features_missing_columns(self):
        data_with_missing_feature = self.sample_data.drop(columns=['GrLivArea'])
        with self.assertRaises(KeyError):
            self.feature_engineer.create_features(data_with_missing_feature)

    @patch('pandas.read_csv')
    def test_load_data(self, mock_read_csv):
        mock_read_csv.return_value = self.sample_data
        result = load_data('dummy_path.csv')
        expected_columns = ['GrLivArea', 'TotalBsmtSF', 'YrSold', 'YearRemodAdd', 'YearBuilt', 
                            'GarageYrBlt', 'FullBath', 'HalfBath', 'BsmtFullBath', 'BsmtHalfBath']
        self.assertEqual(result.columns.tolist(), expected_columns)


if __name__ == '__main__':
    unittest.main()
