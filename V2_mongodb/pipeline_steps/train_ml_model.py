import pickle
from clearml import Task, Dataset, OutputModel
import pandas as pd
import os
import json
from pathlib import Path
from xgboost import XGBClassifier, plot_tree, plot_importance
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, confusion_matrix, accuracy_score, precision_score, recall_score, \
    f1_score
import matplotlib.pyplot as plt
import seaborn as sns


class TrainModel:
    def __init__(self, dataset_id: str):
        self.task = Task.current_task()
        # self.task.set_base_docker(docker_image="python:3.12")
        self.dataset_id = dataset_id

    def get_training_data(self, dataset_id: str):
        if not os.path.exists(Path.cwd().joinpath("../data", dataset_id)):
            dataset = Dataset.get(
                dataset_id=dataset_id
            ).get_mutable_local_copy(
                target_folder=Path.cwd().joinpath("../data", dataset_id)
            )
            dataset = Path(dataset)
        else:
            dataset = Path.cwd().joinpath("../data", dataset_id)

        data_df = pd.read_csv(Path.joinpath(dataset, "merged_input_data.csv"), index_col="SerialNr")

        return data_df

    def create_train_test_split(self, training_data):
        X = training_data[["col_11", "col_12", "col_13", "col_21", "col_22", "col_23"]]
        y = training_data["Error_Label"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)

        return X_train, X_test, y_train, y_test

    def save_model(self, model, parameters, accuracy, mse, precision, recall, f1_score2):
        model_path = Path.cwd().joinpath("../data", "XGBoost_model.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(model, f)

        self.task.upload_artifact(name="trained_xgboost_model", artifact_object=model_path)

        output_model = OutputModel(task=self.task, name="XGBoost Model", framework="xgboost")
        # output_model = OutputModel(name="XGBoost Model", framework="xgboost")
        output_model.update_weights(weights_filename=str(model_path))
        output_model.update_design(config_dict={
            "model_type": "XGBClassifier",
            "parameters": parameters,
            "accuracy": accuracy,
            "mean_squared_error": mse,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score2
        })

        return output_model.id

    def evaluate_model(self, model, X_test, y_test):
        predictions = model.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)
        mse = mean_squared_error(y_test, predictions)
        precision = precision_score(y_test, predictions, average="macro")
        recall = recall_score(y_test, predictions, average="macro")
        f1_score2 = f1_score(y_test, predictions, average="macro")

        self.task.get_logger().report_single_value(name="Accuracy", value=accuracy)
        self.task.get_logger().report_single_value(name="MSE", value=mse)
        self.task.get_logger().report_single_value(name="Precision", value=precision)
        self.task.get_logger().report_single_value(name="Recall", value=recall)
        self.task.get_logger().report_single_value(name="F1", value=f1_score2)

        return predictions, accuracy, mse, precision, recall, f1_score2

    def train_model(self, X_train, X_test, y_train, y_test):
        parameters = {
            "learning_rate": 0.05,
            "n_estimators": 600,
            "max_depth": 3,
            "seed": 42
        }
        self.task.connect(parameters)

        model = XGBClassifier(**parameters)
        model.fit(X_train, y_train)

        predictions, accuracy, mse, precision, recall, f1_score2 = self.evaluate_model(model, X_test, y_test)
        model_id = self.save_model(model, parameters, accuracy, mse, precision, recall, f1_score2)

        return model, predictions, model_id

    def visualize_results(self, model, predictions, y_test, training_data):
        cm = confusion_matrix(y_test, predictions)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, cmap='Blues', fmt='g', xticklabels=training_data.Error.unique(),
                    yticklabels=training_data.Error.unique())
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Confusion Matrix')
        plt.show()

        plot_importance(model)
        plt.show()

    def manage_training(self):
        training_data = self.get_training_data(self.dataset_id)
        self.task.upload_artifact(name="training_data", artifact_object=training_data)
        X_train, X_test, y_train, y_test = self.create_train_test_split(training_data)
        model, predictions, model_id = self.train_model(X_train, X_test, y_train, y_test)
        self.visualize_results(model, predictions, y_test, training_data)

        return model_id


if __name__ == "__main__":
    model_id = TrainModel("baca77c43a6c4201a9189bcaac4e2d39").manage_training()
    # model_id = manage_training("baca77c43a6c4201a9189bcaac4e2d39")
    print(model_id)
