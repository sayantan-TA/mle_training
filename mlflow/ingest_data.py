import os
import tarfile
import urllib.request
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder

import mlflow

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = "/home/sayantanchakrab/mle_training/data/raw/"
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

    housing_cat = data[["ocean_proximity"]]
    data = data.join(pd.get_dummies(housing_cat, drop_first=True))

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
    housing_url=HOUSING_URL,
    housing_path=HOUSING_PATH,
    output_folder: str = "/home/sayantanchakrab/mle_training/data/processed/",
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
    with mlflow.start_run() as run:
        mlflow.log_params(
            {
                "Housing Dataset URL": housing_url,
                "Housing Path": housing_path,
                "Output Folder": output_folder,
            }
        )

        fetch_housing_data()
        housing_data = load_housing_data()

        dataset_prepared = preprocess_data(housing_data)
        train_set, test_set = stratified_split(dataset_prepared)

        train_output_path = os.path.join(output_folder, "train.csv")
        test_output_path = os.path.join(output_folder, "test.csv")

        train_set.to_csv(train_output_path, index=False)
        test_set.to_csv(test_output_path, index=False)

        mlflow.log_artifact(train_output_path, "processed_data")
        mlflow.log_artifact(test_output_path, "processed_data")
        print(f"Storing Processed Data to {output_folder}")


if __name__ == "__main__":
    processed_data()
