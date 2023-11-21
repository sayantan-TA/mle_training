import argparse
import logging

import pandas as pd

from HousePricePrediction.ingest_data import processed_data
from HousePricePrediction.logger import setup_logger
from HousePricePrediction.score import score_model
from HousePricePrediction.train import (
    train_decision_tree,
    train_linear_regression,
    train_random_forest_1,
    train_random_forest_2,
)


def train_models():
    """
    Train machine learning models using the Housing dataset.

    This function trains the following models:
    - Decision Tree
    - Linear Regression
    - Random Forest (with RandomizedSearchCV)
    - Random Forest (with GridSearchCV)

    The trained models are saved for later use.

    Example:
    >>> train_models()
    """
    training_data = pd.read_csv(r"/home/kiit/mle_training/data/processed/train.csv")
    train_decision_tree(training_data)
    logger.info("Decision Tree Model Trained")
    train_linear_regression(training_data)
    logger.info("Linear Regression Model Trained")
    train_random_forest_1(training_data)
    logger.info("Random Forest Model 1 trained")
    train_random_forest_2(training_data)
    logger.info("Random Forest Model 2 trained")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="House Price Prediction Workflow")
    parser.add_argument(
        "--workflow",
        choices=["ingest", "train", "score"],
        required=True,
        help="Specify the workflow step (ingest, train, or score).",
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
        log_file_path=r"/home/kiit/mle_training/logs/main_log.log",
        console_log=not args.no_console_log,
    )
    if args.workflow == "ingest":
        processed_data()
        logger.info("Ingest data workflow completed.")

    elif args.workflow == "train":
        train_models()
        logger.info("Training models workflow completed.")

    elif args.workflow == "score":
        score_model()
        logger.info("Scoring models workflow completed.")
