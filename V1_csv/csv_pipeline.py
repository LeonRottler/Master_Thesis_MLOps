from clearml import Dataset
from clearml.automation.controller import PipelineDecorator
import pandas as pd
import os
from sklearn import preprocessing
from xgboost import XGBClassifier, plot_tree, plot_importance
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns


@PipelineDecorator.component(return_values=["df_station_a", "df_station_b", "df_station_c"], cache=True)
def download_datasets(dataset_id: str):
    dataset = Dataset.get(dataset_id=dataset_id).get_local_copy()
    df_station_a = pd.read_csv(os.path.join(dataset, "StationA.csv"), index_col="SerialNr")
    df_station_b = pd.read_csv(os.path.join(dataset, "StationB.csv"), index_col="SerialNr")
    df_station_c = pd.read_csv(os.path.join(dataset, "StationC.csv"), index_col="SerialNr")

    return df_station_a, df_station_b, df_station_c


@PipelineDecorator.component(return_values=["df_merged"], cache=True)
def merge_datasets(df_station_a: pd.DataFrame, df_station_b: pd.DataFrame, df_station_c: pd.DataFrame):
    df_temp = df_station_a.merge(df_station_b, on="SerialNr", how="outer")
    return df_temp.merge(df_station_c, on="SerialNr", how="outer")


@PipelineDecorator.component(return_values=["df_transformed"], cache=True)
def transform_data(df_merged: pd.DataFrame):
    label_encoder = preprocessing.LabelEncoder()
    df_merged["Error_Label"] = label_encoder.fit_transform(df_merged["Error"])
    return df_merged


@PipelineDecorator.component(return_values=["accuracy", "mse", "model"], cache=True)
def train_model(df: pd.DataFrame):
    X = df[["col_11", "col_12", "col_13", "col_21", "col_22", "col_23"]]
    y = df["Error_Label"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)

    parameters = {
        "learning_rate": 0.05,
        "n_estimators": 600,
        "max_depth": 3,
        "seed": 42
    }
    # task.connect(parameters)

    model = XGBClassifier(**parameters)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    # task.get_logger().report_single_value(name="Accuracy", value=accuracy_score(y_test, predictions))
    # task.get_logger().report_single_value(name="MSE", value=mean_squared_error(y_test, predictions))
    accuracy = accuracy_score(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)

    cm = confusion_matrix(y_test, predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='g', xticklabels=df.Error.unique(),
                yticklabels=df.Error.unique())
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label 1')
    plt.title('Confusion Matrix')
    plt.show()

    plot_importance(model)
    plt.show()

    return accuracy, mse, model


@PipelineDecorator.pipeline(name="ETL Pipline", project="Thesis", version="0.1.0")
def main(dataset_id: str):
    df_station_a, df_station_b, df_station_c = download_datasets(dataset_id)
    df_merged = merge_datasets(df_station_a, df_station_b, df_station_c)
    df_transformed = transform_data(df_merged)
    train_model(df_transformed)


if __name__ == "__main__":
    PipelineDecorator.set_default_execution_queue("services")
    # PipelineDecorator.run_locally()
    main("57c0b3078cad4d0cba2a09800f06f4bb")
    # main("57c0b3078cad4d0cba2a09800f06f4bb")
