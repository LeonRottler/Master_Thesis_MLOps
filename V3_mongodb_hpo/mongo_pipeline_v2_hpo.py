import os
from pymongo import MongoClient
from clearml import Task, Dataset, PipelineDecorator, OutputModel
from clearml.automation import UniformParameterRange, UniformIntegerParameterRange, DiscreteParameterRange, \
    HyperParameterOptimizer, GridSearch, RandomSearch
import json
import pandas as pd
from bson import ObjectId
from pathlib import Path
from sklearn import preprocessing
from xgboost import XGBClassifier, plot_tree, plot_importance
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, confusion_matrix, accuracy_score, precision_score, recall_score, \
    f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import pickle


@PipelineDecorator.component(return_values=["dataset_id"], cache=False)
def extract_data_from_mongodb(database: str, collection: str, id: str, connection_string: str) -> str:
    class DataExtraction:
        def __init__(self, database: str, collection: str, id: str, connection_string: str):
            self.task = Task.current_task()
            self.client = MongoClient(connection_string)
            self.database = database
            self.collection = collection
            self.id = id

        def save_config(self, database: str, collection: str, id: str):
            print(id)
            configuration = {
                "db": database,
                "collection": collection,
                "id": id
            }

            self.task.connect(configuration)

        def json_converter(self, o):
            if isinstance(o, ObjectId):
                return str(o)
            raise TypeError("Type not serializable")

        def get_mongo_db_data(self, database: str, collection: str, id: str) -> list:
            db = self.client[database]
            collection = db[collection]

            return list(collection.find({"id": id}))

        def create_clearml_dataset(self, data: list, id: str) -> str:
            print(id)
            dataset = Dataset.create(
                dataset_name=str(id),
                dataset_project="Thesis",
                dataset_tags=["raw"]
            )

            data_file = "data.json"
            try:
                with open(data_file, "w") as f:
                    json.dump(data, f, default=str)

                dataset.add_files(data_file)
                dataset.finalize(auto_upload=True)

                return dataset.id

            finally:
                if os.path.exists(data_file):
                    os.remove(data_file)

        def extract_data_from_db(self) -> str:
            self.save_config(self.database, self.collection, self.id)
            data = self.get_mongo_db_data(self.database, self.collection, self.id)
            self.client.close()
            return self.create_clearml_dataset(data, id)

    dataset_id = DataExtraction(database, collection, id, connection_string).extract_data_from_db()
    return dataset_id


@PipelineDecorator.component(return_values=["dataset_id"], cache=False)
def process_raw_data(dataset_id_input: str) -> str:
    class DataPreprocessing:
        def __init__(self, dataset_id: str):
            self.task = Task.current_task()
            self.dataset_id = dataset_id

        def get_json_dataset(self, dataset_id: str):
            if not os.path.exists(Path.cwd().joinpath("../data", dataset_id)):
                dataset = Dataset.get(
                    dataset_id=dataset_id
                ).get_mutable_local_copy(
                    target_folder=Path.cwd().joinpath("../data", dataset_id)
                )
                dataset = Path(dataset)
            else:
                dataset = Path.cwd().joinpath("../data", dataset_id)

            file_path = Path.joinpath(dataset, "data.json")
            with open(file_path, "r") as f:
                json_dataset = json.load(f)

            return json_dataset

        def merge_dfs(self, dfs: list):
            merged_df = dfs[0]
            for df in dfs[1:]:
                merged_df = pd.merge(merged_df, df, on="SerialNr", how="outer")

            return merged_df

        def convert_json_to_dataframe(self, dataset_id: str):
            json_dataset = self.get_json_dataset(dataset_id)
            test_stations = list({item for item in json_dataset[0]["test_stations"]})
            test_stations.sort()

            dfs = []
            for station in test_stations:
                df = pd.DataFrame(json_dataset[0]["test_stations"][station])
                df.set_index("SerialNr", inplace=True)
                dfs.append(df)

            return self.merge_dfs(dfs)

        def label_encode_error_column(self, merged_df: pd.DataFrame) -> pd.DataFrame:
            label_encoder = preprocessing.LabelEncoder()
            merged_df["Error_Label"] = label_encoder.fit_transform(merged_df["Error"])
            return merged_df

        def update_dataset_in_clearnl(self, merged_df: pd.DataFrame, dataset_id: str):
            self.task.upload_artifact(name="processed_input_data", artifact_object=merged_df)

            dataset = Dataset.create(
                dataset_name="processed_dataset",
                dataset_project="Thesis",
                dataset_tags=["processed"],
                parent_datasets=[dataset_id]
            )

            data_dir = Path.cwd().joinpath("../data", dataset_id)

            if os.path.exists(data_dir):
                merged_df.to_csv(Path.joinpath(data_dir, "merged_input_data.csv"))
                dataset.sync_folder(data_dir)
                dataset.finalize(auto_upload=True)

                return dataset.id

        def manage_preprocessing(self):
            merged_df = self.convert_json_to_dataframe(self.dataset_id)
            merged_df = self.label_encode_error_column(merged_df)
            updated_dataset_id = self.update_dataset_in_clearnl(merged_df, self.dataset_id)
            return updated_dataset_id

    dataset_id = DataPreprocessing(dataset_id_input).manage_preprocessing()
    return dataset_id


@PipelineDecorator.component(return_values=["model_id"], cache=False)
def train_model(dataset_id: str, hyper_parameter: dict = None) -> str:
    class TrainModel:
        def __init__(self, dataset_id: str, hyper_parameter: dict = None):
            self.task = Task.current_task()
            self.dataset_id = dataset_id
            self.hyper_parameter = hyper_parameter

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
            X = training_data.filter(regex="col_")
            y = training_data["Error_Label"]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)

            return X_train, X_test, y_train, y_test

        def save_model(self, model, parameters, accuracy, mse, precision, recall, f1_score2):
            model_path = Path.cwd().joinpath("../data", "XGBoost_model.pkl")
            with open(model_path, "wb") as f:
                pickle.dump(model, f)

            self.task.upload_artifact(name="trained_xgboost_model", artifact_object=model_path)

            model_path2 = Path.cwd().joinpath("data", "XGBoost_model")
            model.save_model(model_path2)

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

        def convert_hyper_parameter_dict_value(self, value):
            try:
                return float(value) if "." in value else int(value)
            except ValueError:
                return value

        def train_model(self, X_train, X_test, y_train, y_test):
            if self.hyper_parameter is not None:
                modified_hyper_parameter = {key.replace('General/', ''): self.convert_hyper_parameter_dict_value(value)
                                            for key, value in self.hyper_parameter.items()}
                print(modified_hyper_parameter)
                self.task.connect(modified_hyper_parameter)

                model = XGBClassifier(**modified_hyper_parameter)
                model.fit(X_train, y_train)

                predictions, accuracy, mse, precision, recall, f1_score2 = self.evaluate_model(model, X_test, y_test)
                model_id = self.save_model(model, modified_hyper_parameter, accuracy, mse, precision, recall, f1_score2)

                return model, predictions, model_id
            else:
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
            if self.hyper_parameter is not None:
                training_data = self.get_training_data(self.dataset_id)
                self.task.upload_artifact(name="training_data", artifact_object=training_data)
                X_train, X_test, y_train, y_test = self.create_train_test_split(training_data)
                model, predictions, model_id = self.train_model(X_train, X_test, y_train, y_test)
                self.visualize_results(model, predictions, y_test, training_data)
                return model_id, self.task.id
            else:
                training_data = self.get_training_data(self.dataset_id)
                self.task.upload_artifact(name="training_data", artifact_object=training_data)
                X_train, X_test, y_train, y_test = self.create_train_test_split(training_data)
                model, predictions, model_id = self.train_model(X_train, X_test, y_train, y_test)
                self.visualize_results(model, predictions, y_test, training_data)
                return model_id, self.task.id

    model_id, task_id = TrainModel(dataset_id, hyper_parameter).manage_training()
    return model_id, task_id


@PipelineDecorator.component(return_values=["model_id"], cache=False)
def hyperparameter_optimization(train_task_id):
    class HPO:
        def __init__(self, train_task_id):
            self.train_task_id = train_task_id

        def run_optimizer(self):
            optimizer = HyperParameterOptimizer(
                base_task_id=self.train_task_id,
                hyper_parameters=[
                    UniformParameterRange("General/learning_rate", min_value=0.001, max_value=0.2),
                    UniformIntegerParameterRange("General/max_depth", min_value=1, max_value=5),
                    UniformIntegerParameterRange("General/n_estimators", min_value=400, max_value=800)
                ],
                objective_metric_title="F1",
                objective_metric_series="f1_score2",
                objective_metric_sign="max",

                execution_queue="services",
                optimization_time_limit=1.0,

                optimizer_class=RandomSearch
            )

            optimizer.set_report_period(1)
            optimizer.start()
            optimizer.set_time_limit(1.0)
            optimizer.wait()
            best_experiment = optimizer.get_top_experiments_details(1)
            optimizer.stop()

            return best_experiment

    best_experiment = HPO(train_task_id).run_optimizer()
    hyper_parameters = best_experiment[0]["hyper_parameters"]
    return best_experiment, hyper_parameters


@PipelineDecorator.pipeline(name="MongoDB Pipline", project="Thesis", version="0.3.0")
def run_pipeline(database: str, collection: str, id: str, connection_string: str):
    dataset_id = extract_data_from_mongodb(database, collection, id, connection_string)
    dataset_id = process_raw_data(dataset_id)
    model_id, train_task_id = train_model(dataset_id)
    best_experiment, hyper_parameters = hyperparameter_optimization(train_task_id)
    model_id, train_task_id = train_model(dataset_id, hyper_parameters)


if __name__ == "__main__":
    # PipelineDecorator.set_default_execution_queue("services")
    PipelineDecorator.run_locally()
    # local connection string: mongodb://leon:thesis2024!@localhost:27017/
    # remote connection string: mongodb+srv://leon:<password>@thesis.4kmqr.mongodb.net/?retryWrites=true&w=majority&appName=Thesis
    run_pipeline(database="PQM", collection="thesis", id="production_line_1",
                 connection_string="mongodb://leon:thesis2024!@localhost:27017/")
