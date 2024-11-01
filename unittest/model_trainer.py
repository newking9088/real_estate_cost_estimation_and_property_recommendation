import unittest
import numpy as np
import pandas as pd
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from collections import defaultdict
import time

class ModelTrainer:
    def __init__(self, models):
        self.models = models
        self.scores = defaultdict(list)

    def train_and_evaluate(self, X_train, y_train, X_test, y_test):
        for name, model in self.models.items():
            start_time = time.time()
            model.fit(X_train, y_train)
            end_time = time.time()
            self.evaluate_model(name, model, X_train, y_train, X_test, y_test, end_time - start_time)

    def evaluate_model(self, name, model, X_train, y_train, X_test, y_test, train_time):
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        self.scores['model'].append(name)
        self.scores['train_time_sec'].append(train_time)
        self.scores['train_r2'].append(r2_score(y_train, y_train_pred))
        self.scores['test_r2'].append(r2_score(y_test, y_test_pred))
        self.scores['train_mae'].append(mean_absolute_error(y_train, y_train_pred))
        self.scores['test_mae'].append(mean_absolute_error(y_test, y_test_pred))


class TestModelTrainer(unittest.TestCase):
    def setUp(self):
        # Create a synthetic dataset
        self.X, self.y = make_regression(n_samples=100, n_features=5, noise=0.1, random_state=42)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)

        # Initialize models
        self.models = {
            'Linear Regression': LinearRegression(),
            'Random Forest': RandomForestRegressor(random_state=42)
        }

        # Initialize the ModelTrainer
        self.model_trainer = ModelTrainer(self.models)

    def test_train_and_evaluate(self):
        # Train and evaluate the models
        self.model_trainer.train_and_evaluate(self.X_train, self.y_train, self.X_test, self.y_test)

        # Check that scores are recorded for each model
        self.assertEqual(len(self.model_trainer.scores['model']), len(self.models))

        # Check that training times are recorded
        self.assertTrue(all(isinstance(time, float) for time in self.model_trainer.scores['train_time_sec']))

        # Check that R2 scores are in the expected range
        for train_r2, test_r2 in zip(self.model_trainer.scores['train_r2'], self.model_trainer.scores['test_r2']):
            self.assertGreaterEqual(train_r2, 0)
            self.assertGreaterEqual(test_r2, 0)

        # Check that MAE scores are non-negative
        for train_mae, test_mae in zip(self.model_trainer.scores['train_mae'], self.model_trainer.scores['test_mae']):
            self.assertGreaterEqual(train_mae, 0)
            self.assertGreaterEqual(test_mae, 0)

if __name__ == '__main__':
    unittest.main()
