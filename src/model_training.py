import os
import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV
from src.logger import get_logger
from src.custom_exception import CustomException

import mlflow
import mlflow.sklearn

logger = get_logger(__name__)

class ModelTraining:
    def __init__(self, processed_data_path="artifacts/processed"):
        self.processed_data_path = processed_data_path
        self.model_dir = "artifacts/model"
        os.makedirs(self.model_dir, exist_ok=True)

        logger.info("Model Training Initialization...")

    def load_data(self):
        try:
            self.X_train = joblib.load(os.path.join(self.processed_data_path, "X_train.pkl"))
            self.X_test = joblib.load(os.path.join(self.processed_data_path, "X_test.pkl"))
            self.y_train = joblib.load(os.path.join(self.processed_data_path, "y_train.pkl"))
            self.y_test = joblib.load(os.path.join(self.processed_data_path, "y_test.pkl"))

            logger.info("Data loaded for Model")
        except Exception as e:
            logger.error(f"Error while loading data for model..{e}")
            raise CustomException("Failed to load data for model..", e)

    def train_model(self):
        try:
            param_grid = {
                'n_estimators': [100, 150],
                'learning_rate': [0.05, 0.1],
                'max_depth': [3, 4]
            }

            base_model = GradientBoostingClassifier(random_state=42)
            grid_search = GridSearchCV(
                estimator=base_model,
                param_grid=param_grid,
                cv=3,
                n_jobs=-1,
                scoring='accuracy',
                verbose=1
            )

            grid_search.fit(self.X_train, self.y_train)
            self.model = grid_search.best_estimator_

            logger.info(f"Best Params: {grid_search.best_params_}")
            logger.info(f"Best Cross-Validation Score: {grid_search.best_score_}")

            joblib.dump(self.model, os.path.join(self.model_dir, "model.pkl"))
            logger.info("Model trained and saved successfully...")
        except Exception as e:
            logger.error(f"Error while training the model {e}")
            raise CustomException("Failed to train the model...", e)

    def evaluate_model(self):
        try:
            y_pred = self.model.predict(self.X_test)
            y_proba = self.model.predict_proba(self.X_test)[:, 1] if len(self.y_test.unique()) == 2 else None

            accuracy = accuracy_score(self.y_test, y_pred)
            precision = precision_score(self.y_test, y_pred, average = "weighted")
            recall = recall_score(self.y_test, y_pred, average = "weighted")
            f1 = f1_score(self.y_test, y_pred, average = "weighted")

            mlflow.log_metric("Accuracy",accuracy) # type: ignore
            mlflow.log_metric("Precision",precision) # type: ignore
            mlflow.log_metric("Recall Score", recall) # type: ignore
            mlflow.log_metric("F1_Score",f1) # type: ignore


            logger.info(f"Accuracy: {accuracy}; Precision: {precision}; Recall: {recall}; f1_score: {f1}")

            if y_proba is not None:
                roc_auc = roc_auc_score(self.y_test.map({'No': 0, 'Yes': 1}), y_proba)
                mlflow.log_metric("ROC-AUC", roc_auc) # type: ignore
                logger.info(f"ROC AUC Score: {roc_auc}")

            logger.info("Model Evaluation done...")
        except Exception as e:
            logger.error(f"Error while evaluating the model {e}")
            raise CustomException("Failed to evaluate the model...", e)

    def run(self):
        self.load_data()
        self.train_model()
        self.evaluate_model()

if __name__ == "__main__":
    with mlflow.start_run():
        trainer = ModelTraining()
        trainer.run()
    