import unittest

import pandas as pd

from HousePricePrediction.train import (
    train_decision_tree,
    train_linear_regression,
    train_random_forest_1,
    train_random_forest_2,
)


class TestModelTraining(unittest.TestCase):
    def setUp(self):
        # sample DataFrame
        data = {
            "median_income": [1.0, 2.0, 3.0, 4.0, 5.0],
            "total_rooms": [100, 200, 300, 400, 500],
            "households": [10, 20, 30, 40, 50],
            "total_bedrooms": [8, 15, 25, 35, 45],
            "population": [50, 100, 150, 200, 250],
            "ocean_proximity": [0, 1, 0, 1, 0],
            "median_house_value": [200000, 250000, 300000, 350000, 400000],
        }
        self.training_data = pd.DataFrame(data)

    def test_train_random_forest_1(self):
        model = train_random_forest_1(self.training_data)
        self.assertIsNotNone(model)

    def test_train_random_forest_2(self):
        model = train_random_forest_2(self.training_data)
        self.assertIsNotNone(model)

    def test_train_linear_regression(self):
        model = train_linear_regression(self.training_data)
        self.assertIsNotNone(model)

    def test_train_decision_tree(self):
        model = train_decision_tree(self.training_data)
        self.assertIsNotNone(model)


if __name__ == "__main__":
    unittest.main()
