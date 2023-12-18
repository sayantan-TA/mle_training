import subprocess

import mlflow

if __name__ == "__main__":
    with mlflow.start_run(run_name="House_Price_Prediction_Workflow") as main_run:
        with mlflow.start_run(
            run_name="Data_Preparation", nested=True
        ) as data_prep_run:
            subprocess.run(["python", "ingest_data.py"])
            # mlflow.log_params({"Data Preperation": "Done"})

        with mlflow.start_run(run_name="Model_Training", nested=True) as training_run:
            subprocess.run(["python", "train.py"])
            # mlflow.log_params(
            #     {
            #         "Random Forest Model 1 & 2": "Done",
            #         "Linear Regression": "Done",
            #         "Decision Tree": "Done",
            #     }
            # )

        with mlflow.start_run(run_name="Model_Scoring", nested=True) as scoring_run:
            subprocess.run(["python", "score.py"])
            # mlflow.log_params({"scoring": "Done"})
