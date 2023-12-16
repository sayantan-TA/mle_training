import os
import pickle

import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

import mlflow


def load_model(model_folder: str, model_name: str):
    """
    Load a machine learning model from the specified folder.

    Parameters:
    - model_folder (str): Path to the folder containing the model pickle file.
    - model_name (str): Name of the model (without extension).

    Returns:
    - Any: Loaded machine learning model.

    Example:
    >>>loaded_model = load_model(model_folder="/path/to/models", model_name="random_forest")
    """
    model_path = os.path.join(model_folder, f"{model_name}_model.pkl")
    with open(model_path, "rb") as model_file:
        model = pickle.load(model_file)
    return model


def load_data(dataset_folder: str) -> pd.DataFrame:
    """
    Load the testing dataset from the specified folder.

    Parameters:
    - dataset_folder (str): Path to the folder containing the testing dataset.

    Returns:
    - pd.DataFrame: Loaded testing dataset.

    Example:
    >>> test_data = load_data(dataset_folder="/path/to/data")
    """
    test_path = os.path.join(dataset_folder, "test.csv")
    return pd.read_csv(test_path)


def score_models(model, X_test, y_test):
    """
    Calculate Mean Squared Error (MSE) and Root Mean Squared Error (RMSE) for a regression model.

    Parameters:
    - model: Trained regression model.
    - X_test: Features of the testing dataset.
    - y_test: Target values of the testing dataset.

    Returns:
    - Tuple: A tuple containing MSE and RMSE.

    Example:
    >>> mse, rmse = score_models(model, X_test, y_test)
    """
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    rmse = mean_absolute_error(y_test, predictions)
    return mse, rmse


def score_model():
    """
    Score each trained regression model on the testing dataset and print the MSE and RMSE.

    Models loaded for scoring:
    1. Random Forest Model 1
    2. Random Forest Model 2
    3. Linear Regression Model
    4. Decision Tree Model

    Scores are saved to a text file named 'scores.txt' in the 'outputs' folder.

    Example:
    >>> score_model()
    """
    test_path = r"/home/sayantanchakrab/mle_training/data/processed/test.csv"
    test_data = pd.read_csv(test_path)
    X_test = test_data.drop("median_house_value", axis=1)
    y_test = test_data["median_house_value"]

    with mlflow.start_run() as run:
        rf_model = load_model(
            r"/home/sayantanchakrab/mle_training/pickles/",
            "House_Price_Prediction Random_Forest_1",
        )
        rf_mse, rf_rmse = score_models(rf_model, X_test, y_test)
        # mlflow.log_params({"Model_Name": "Random Forest Model 1"})
        mlflow.log_metrics({"MSE": rf_mse, "RMSE": rf_rmse})
        print(f"Random Forest Model 1- MSE: {rf_mse}, RMSE: {rf_rmse}")

        rf_model_2 = load_model(
            r"/home/sayantanchakrab/mle_training/pickles/",
            "House_Price_Prediction Random_Forest_2",
        )
        rf_mse_2, rf_rmse_2 = score_models(rf_model_2, X_test, y_test)
        # mlflow.log_params({"Model_Name": "Random Forest Model 2"})
        mlflow.log_metrics({"MSE": rf_mse_2, "RMSE": rf_rmse_2})
        print(f"Random Forest Model 2- MSE: {rf_mse_2}, RMSE: {rf_rmse_2}")

        lr_model = load_model(
            r"/home/sayantanchakrab/mle_training/pickles/",
            "House_Price_Prediction Linear Regression",
        )
        lr_mse, lr_rmse = score_models(lr_model, X_test, y_test)
        # mlflow.log_params({"Model_Name": "Linear Regression Model"})
        mlflow.log_metrics({"MSE": lr_mse, "RMSE": lr_rmse})
        print(f"Linear Regression Model - MSE: {lr_mse}, RMSE: {lr_rmse}")

        dt_model = load_model(
            r"/home/sayantanchakrab/mle_training/pickles/",
            "House_Price_Prediction Decision Tree",
        )
        dt_mse, dt_rmse = score_models(dt_model, X_test, y_test)
        # mlflow.log_params({"Model_Name": "Decision Tree Model"})
        mlflow.log_metrics({"MSE": dt_mse, "RMSE": dt_rmse})
        print(f"Decision Tree Model - MSE: {dt_mse}, RMSE: {dt_rmse}")

        os.makedirs(r"/home/sayantanchakrab/mle_training/outputs/", exist_ok=True)
        with open(
            os.path.join(r"/home/sayantanchakrab/mle_training/outputs/", "scores.txt"),
            "w",
        ) as scores_file:
            scores_file.write(
                f"Random Forest Model 1- MSE: {rf_mse}, RMSE: {rf_rmse}\n"
            )
            scores_file.write(
                f"Random Forest Model 2- MSE: {rf_mse_2}, RMSE: {rf_rmse_2}\n"
            )
            scores_file.write(
                f"Linear Regression Model - MSE: {lr_mse}, RMSE: {lr_rmse}\n"
            )
            scores_file.write(f"Decision Tree Model - MSE: {dt_mse}, RMSE: {dt_rmse}\n")
    return


if __name__ == "__main__":
    score_model()
