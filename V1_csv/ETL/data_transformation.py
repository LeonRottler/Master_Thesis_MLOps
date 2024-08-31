from clearml import Task, Dataset
import pandas as pd
import os
from pathlib import Path
from sklearn import preprocessing

task = Task.init(project_name="Thesis", task_name="Data Transformation")

data_dir = Path.cwd().parent.joinpath("data")

#  Getting the station data from clearml data storage and transform it into pandas dataframe
if not os.path.exists(data_dir.joinpath("792eaa29a3b74817b76b8a609286936d")):
    dataset = Dataset.get(
        dataset_id="792eaa29a3b74817b76b8a609286936d"
    ).get_mutable_local_copy(
        target_folder=Path.cwd().parent.joinpath("data", "792eaa29a3b74817b76b8a609286936d")
    )

#
df_stationA = pd.read_csv(data_dir.joinpath("792eaa29a3b74817b76b8a609286936d", "StationA.csv"),
                          index_col="SerialNr")
df_stationB = pd.read_csv(data_dir.joinpath("792eaa29a3b74817b76b8a609286936d", "StationB.csv"),
                          index_col="SerialNr")
df_stationC = pd.read_csv(data_dir.joinpath("792eaa29a3b74817b76b8a609286936d", "StationC.csv"),
                          index_col="SerialNr")

# merging the different dataframes
df_temp = df_stationA.merge(df_stationB, on="SerialNr", how="outer")
df_merged = df_temp.merge(df_stationC, on="SerialNr", how="outer")

df_merged.to_csv(data_dir.joinpath("792eaa29a3b74817b76b8a609286936d", "StationsMerged.csv"))

task.upload_artifact(name="StationsMergedDF", artifact_object=df_merged)
task.upload_artifact(name="StationsMerged.csv", artifact_object=data_dir.joinpath("792eaa29a3b74817b76b8a609286936d",
                                                                                  "StationsMerged.csv"))

label_encoder = preprocessing.LabelEncoder()

df_merged["Error_Label"] = label_encoder.fit_transform(df_merged["Error"])

df_merged.to_csv(data_dir.joinpath("792eaa29a3b74817b76b8a609286936d", "Stations_Merged_Encoded.csv"))
task.upload_artifact(name="Stations_Merged_Encoded_DF", artifact_object=df_merged)
task.upload_artifact(name="Stations_Merged_Encoded.csv", artifact_object=data_dir.joinpath(
    "792eaa29a3b74817b76b8a609286936d", "Stations_Merged_Encoded.csv"))

dataset = Dataset.create(
    dataset_project="Thesis",
    dataset_name="Stations_Merged_Encoded",
    parent_datasets=["792eaa29a3b74817b76b8a609286936d"]
)
dataset.sync_folder(data_dir.joinpath("792eaa29a3b74817b76b8a609286936d"))
dataset.finalize(auto_upload=True)

"""

def save_file_in_clearml_dataset(file_path: Path, dataset_name: str):
    dataset = Dataset.create(
        dataset_project="Thesis", dataset_name=dataset_name
    )
    dataset.add_files(file_path)
    dataset.upload()
    dataset.finalize()

dataset_station_a = Dataset.get(dataset_project="Thesis", dataset_name="StationA.csv").get_local_copy()
df_stationA = pd.read_csv(os.path.join(dataset_station_a, "StationA.csv"), index_col="SerialNr")

dataset_station_b = Dataset.get(dataset_project="Thesis", dataset_name="StationB.csv").get_local_copy()
df_stationB = pd.read_csv(os.path.join(dataset_station_b, "StationB.csv"), index_col="SerialNr")

dataset_station_c = Dataset.get(dataset_project="Thesis", dataset_name="StationC.csv").get_local_copy()
df_stationC = pd.read_csv(os.path.join(dataset_station_c, "StationC.csv"), index_col="SerialNr")

# merging the different dataframes
df_temp = df_stationA.merge(df_stationB, on="SerialNr", how="outer")
df_merged = df_temp.merge(df_stationC, on="SerialNr", how="outer")

file_path = Path.cwd().parent.joinpath("data", "StationsMerged.csv")

df_merged.to_csv(file_path)

task.upload_artifact(name="StationsMergedDF", artifact_object=df_merged)
task.upload_artifact(name="StationsMerged.csv", artifact_object=file_path)
save_file_in_clearml_dataset(file_path, "StationsMerged.csv")"""

"""
label_encoder = preprocessing.LabelEncoder()

df_merged.Error = label_encoder.fit_transform(df_merged.Error)
file_path_transformed = Path.cwd().parent.joinpath("data", "StationsMerged_transformed.csv")

task.upload_artifact(name="StationsMergedDF_Transformed", artifact_object=df_merged)
task.upload_artifact(name="StationsMerged_Transformed.csv", artifact_object=file_path)
save_file_in_clearml_dataset(file_path, "StationsMerged_Transformed.csv")"""
