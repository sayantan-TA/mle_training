import argparse
import logging
import os
import pickle

import pandas as pd
from scipy.stats import randint
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.tree import DecisionTreeRegressor

from HousePricePrediction.logger import setup_logger


def load_data(input_folder):
    """
    Load the preprocessed training dataset.

    Parameters:
    - input_folder (str): Path to the folder containing the preprocessed training dataset.

    Returns:
    - pd.DataFrame: Loaded training dataset.

    Example:
    >>> load_data("/path/to/training_data_folder")
    """
    train_path = os.path.join(input_folder, "train.csv")
    return pd.read_csv(train_path)


def train_random_forest_1(training_data):
    """
    Train a Random Forest model using hyperparameter tuning.

    Parameters:
    - training_data (pd.DataFrame): The training dataset.

    Returns:
    - RandomForestRegressor: Trained Random Forest model.

    Example:
    >>> train_random_forest_1(training_data)
    """
    X_train = training_data.drop("median_house_value", axis=1)
    y_train = training_data["median_house_value"]
    param_distribs = {
        "n_estimators": randint(low=1, high=200),
        "max_features": randint(low=1, high=8),
    }

    forest_reg = RandomForestRegressor(random_state=42)
    rnd_search = RandomizedSearchCV(
        forest_reg,
        param_distributions=param_distribs,
        n_iter=10,
        cv=5,
        scoring="neg_mean_squared_error",
        random_state=42,
    )
    rnd_search.fit(X_train, y_train)
    save_model(rnd_search.best_estimator_, "House_Price_Prediction Random_Forest_1")
    return rnd_search.best_estimator_


def train_random_forest_2(training_data):
    """
    Train a Random Forest model with cross-validation and hyperparameter tuning.

    Parameters:
    - training_data (pd.DataFrame): The training dataset.

    Returns:
    - RandomForestRegressor: Trained Random Forest model.

    Example:
    >>> train_random_forest_2(training_data)
    """
    X_train = training_data.drop("median_house_value", axis=1)
    y_train = training_data["median_house_value"]
    param_grid = [
        # try 12 (3×4) combinations of hyperparameters
        {"n_estimators": [3, 10, 30], "max_features": [2, 4, 6, 8]},
        # then try 6 (2×3) combinations with bootstrap set as False
        {"bootstrap": [False], "n_estimators": [3, 10], "max_features": [2, 3, 4]},
    ]
    forest_reg = RandomForestRegressor(random_state=42)
    # train across 5 folds, that's a total of (12+6)*5=90 rounds of training
    grid_search = GridSearchCV(
        forest_reg,
        param_grid,
        cv=5,
        scoring="neg_mean_squared_error",
        return_train_score=True,
    )
    grid_search.fit(X_train, y_train)
    save_model(grid_search.best_estimator_, "House_Price_Prediction Random_Forest_2")
    return grid_search.best_estimator_


def train_linear_regression(training_data):
    """
    Train a Linear Regression model.

    Parameters:
    - training_data (pd.DataFrame): The training dataset.

    Returns:
    - LinearRegression: Trained Linear Regression model.

    Example:
    >>> train_linear_regression(training_data)
    """
    X_train = training_data.drop("median_house_value", axis=1)
    y_train = training_data["median_house_value"]
    lin_reg = LinearRegression()
    lin_reg.fit(X_train, y_train)
    save_model(lin_reg, "House_Price_Prediction Linear Regression")
    return lin_reg


def train_decision_tree(training_data):
    """
    Train a Decision Tree model.

    Parameters:
    - training_data (pd.DataFrame): The training dataset.

    Returns:
    - DecisionTreeRegressor: Trained Decision Tree model.

    Example:
    >>> train_decision_tree(training_data)
    """
    X_train = training_data.drop("median_house_value", axis=1)
    y_train = training_data["median_house_value"]
    tree_reg = DecisionTreeRegressor(random_state=42)
    tree_reg.fit(X_train, y_train)
    save_model(tree_reg, "House_Price_Prediction Decision Tree")
    return tree_reg


def save_model(
    model,
    model_name,
    output_folder=r"/home/kiit/mle_training/pickles/",
):
    """
    Save the trained model as a pickle file.

    Parameters:
    - model: The trained machine learning model.
    - model_name (str): The name of the model.
    - output_folder (str, optional): The folder where the model will be saved. Defaults to
      "/home/kiit/mle_training/pickles/".

    Example:
    >>> save_model(trained_model, "Random_Forest_1", output_folder="/path/to/save/")
    """
    model_path = os.path.join(output_folder, f"{model_name}_model.pkl")
    with open(model_path, "wb") as model_file:
        pickle.dump(model, model_file)
    print(f"{model_name.capitalize()} Model saved to {model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train regression models on the provided dataset."
    )
    parser.add_argument(
        "input_folder",
        type=str,
        help="Input folder path containing the preprocessed training dataset.",
    )
    parser.add_argument(
        "output_folder",
        type=str,
        help="Output folder path for saving the trained models.",
    )
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
        log_file_path=r"/home/kiit/mle_training/logs/train_log.log",
        console_log=not args.no_console_log,
    )

    training_data = load_data(args.input_folder)
    logger.info("Dividing Dataset")

    # Train Random Forest model 1
    rf_model_1 = train_random_forest_1(training_data)
    save_model(rf_model_1, "House_Price_Prediction Random_Forest_1", args.output_folder)
    logger.info("Random Forest Model 1 trained")

    # Train Random Forest model 2
    rf_model_2 = train_random_forest_2(training_data)
    save_model(rf_model_2, "House_Price_Prediction Random_Forest_2", args.output_folder)
    logger.info("Random Forest Model 2 trained")

    # Train Linear Regression model
    lr_model = train_linear_regression(training_data)
    save_model(lr_model, "House_Price_Prediction Linear Regression", args.output_folder)
    logger.info("Linear Regression Model trained")

    # Train Decision Tree model
    dt_model = train_decision_tree(training_data)
    save_model(dt_model, "House_Price_Prediction Decision Tree", args.output_folder)
    logger.info("Decision Tree Model trained")
