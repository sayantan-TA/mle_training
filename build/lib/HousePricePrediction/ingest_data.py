import argparse
import logging
import os
import tarfile
import urllib.request
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder

from HousePricePrediction.logger import setup_logger

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = "/home/kiit/mle_training/data/raw/"
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"


def fetch_housing_data(
    housing_url: str = HOUSING_URL, housing_path: str = HOUSING_PATH
):
    """
    Download and extract the Housing dataset.

    Parameters:
    - housing_url (str): The URL from which to download the dataset.
    - housing_path (str): The local path to store the downloaded and extracted dataset.

    Raises:
    - urllib.error.URLError: If there is an error in downloading the dataset.
    - tarfile.TarError: If there is an error in extracting the dataset.

    Example:
    >>> fetch_housing_data()
    """
    os.makedirs(housing_path, exist_ok=True)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()


def load_housing_data(housing_path: str = HOUSING_PATH) -> pd.DataFrame:
    """
    Load the Housing dataset from a CSV file.

    Parameters:
    - housing_path (str): The local path where the housing dataset CSV file is located.

    Returns:
    - pd.DataFrame: A Pandas DataFrame containing the loaded dataset.

    Example:
    >>> load_housing_data()
    """
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)


def stratified_split(
    data: pd.DataFrame, test_size: float = 0.2, random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Stratified split of the dataset based on median income.

    Parameters:
    - data (pd.DataFrame): The input dataset containing the "median_income" column.
    - test_size (float, optional): The proportion of the dataset to include in the test split.
      Default is 0.2 (20%).
    - random_state (int, optional): Seed for random number generation to ensure reproducibility.
      Default is 42.

    Returns:
    - Tuple[pd.DataFrame, pd.DataFrame]: A tuple containing the training set and the test set.

    Example:
    >>> stratified_split(housing_data)
    """
    data["income_cat"] = pd.cut(
        data["median_income"],
        bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf],
        labels=[1, 2, 3, 4, 5],
    )

    split = StratifiedShuffleSplit(
        n_splits=1, test_size=test_size, random_state=random_state
    )
    for train_index, test_index in split.split(data, data["income_cat"]):
        train_set = data.loc[train_index]
        test_set = data.loc[test_index]

    for set_ in (train_set, test_set):
        set_.drop("income_cat", axis=1, inplace=True)

    return train_set, test_set


def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the input data.

    Parameters:
    - data (pd.DataFrame): The input dataset containing the features.

    Returns:
    - pd.DataFrame: The preprocessed dataset with additional engineered features.

    Example:
    >>> preprocess_data(housing_data)
    """
    data["rooms_per_household"] = data["total_rooms"] / data["households"]
    data["bedrooms_per_room"] = data["total_bedrooms"] / data["total_rooms"]
    data["population_per_household"] = data["population"] / data["households"]

    encoder = LabelEncoder()
    data["ocean_proximity_encoded"] = encoder.fit_transform(data["ocean_proximity"])

    imputer = SimpleImputer(strategy="median")
    housing_num = data.drop("ocean_proximity", axis=1)
    imputer.fit(housing_num)
    X = imputer.transform(housing_num)

    housing_tr = pd.DataFrame(X, columns=housing_num.columns, index=data.index)
    housing_tr["rooms_per_household"] = (
        housing_tr["total_rooms"] / housing_tr["households"]
    )
    housing_tr["bedrooms_per_room"] = (
        housing_tr["total_bedrooms"] / housing_tr["total_rooms"]
    )
    housing_tr["population_per_household"] = (
        housing_tr["population"] / housing_tr["households"]
    )

    return housing_tr


def processed_data(
    output_folder: str = "/home/kiit/mle_training/data/processed/",
) -> None:
    """
    Process the housing data, perform stratified split, preprocess the features,
    and store the prepared datasets into the specified output folder.

    Parameters:
    - output_folder (str): The path to the folder where the processed data will be stored.

    Returns:
    - None

    Example:
    >>> processed_data("/path/to/output_folder/")
    Storing Processed Data to Output Folder
    """

    fetch_housing_data()
    housing_data = load_housing_data()

    train_set, test_set = stratified_split(housing_data)

    train_prepared = preprocess_data(train_set)
    test_prepared = preprocess_data(test_set)

    train_prepared.to_csv(os.path.join(output_folder, "train.csv"), index=False)
    test_prepared.to_csv(os.path.join(output_folder, "test.csv"), index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest data script")
    parser.add_argument("output_folder", help="Output folder path")
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Specify the log level (e.g., DEBUG, INFO, WARNING, ERROR)",
    )
    parser.add_argument(
        "--log-to-file", action="store_true", help="Log messages to a file"
    )
    parser.add_argument(
        "--no-console-log",
        action="store_true",
        help="Do not log messages to the console",
    )
    args = parser.parse_args()
    logger = setup_logger(
        log_level=getattr(logging, args.log_level.upper()),
        log_to_file=args.log_to_file,
        log_file_path=r"/home/kiit/mle_training/logs/ingest_data_log.log",
        console_log=not args.no_console_log,
    )
    processed_data(args.output_folder)

    logger.info("Data Processed")
    logger.info(f"Storing Processed Data to Output Folder: {args.output_folder}")
