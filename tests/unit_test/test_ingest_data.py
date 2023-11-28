import os
import unittest

import pandas as pd

from HousePricePrediction import ingest_data


class TestIngestData(unittest.TestCase):
    def test_load_housing_data(self):
        sample_csv_content = "ID,NAME\n1,Rahul\n2,Rohan"
        sample_csv_path = "tmp/"

        with open(sample_csv_path + "housing.csv", "w") as file:
            file.write(sample_csv_content)

        ingest_data.load_housing_data.HOUSING_PATH = "/tmp/"

        housing_data = ingest_data.load_housing_data(sample_csv_path)

        expected_data = pd.DataFrame({"ID": [1, 2], "NAME": ["Rahul", "Rohan"]})
        pd.testing.assert_frame_equal(housing_data, expected_data)

        # Clean
        os.remove(sample_csv_path + "housing.csv")

    def setUp(self):
        data = {
            "median_income": [1.0, 2.0, 3.0, 4.0, 5.0],
            "total_rooms": [100, 200, 300, 400, 500],
            "households": [10, 20, 30, 40, 50],
            "total_bedrooms": [8, 15, 25, 35, 45],
            "population": [50, 100, 150, 200, 250],
            "ocean_proximity": [0, 1, 0, 1, 0],
            "median_house_value": [200000, 250000, 300000, 350000, 400000],
        }
        self.df = pd.DataFrame(data)

    def test_preprocess_data(self):
        processed_df = ingest_data.preprocess_data(self.df)
        self.assertIsNotNone(processed_df)


if __name__ == "__main__":
    unittest.main()
