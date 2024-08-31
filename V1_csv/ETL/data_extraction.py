from clearml import Task, Dataset
from pathlib import Path
import os

task = Task.init(project_name="Thesis", task_name="Data Extraction")

"""
def extract_data_to_clearml(data_storage_path: str):
    for file in os.listdir(data_storage_path):
        if "Station" in str(file):
            csv_path = os.path.join(data_storage_path, file)
            dataset = Dataset.create(
                dataset_project="Thesis", dataset_name=file.split(".")[0]
            )

            # Create a dataset with ClearML`s Dataset class
            dataset.add_files(csv_path)

            # Upload dataset to ClearML server
            dataset.upload()

            # commit dataset changes
            dataset.finalize()"""

dataset = Dataset.create(
    dataset_project="Thesis",
    dataset_name="StationData"
)

dataset.add_files(Path.cwd().parent.joinpath("data"))
dataset.upload()
dataset.finalize()

# extract_data_to_clearml(Path.cwd().parent.joinpath("data"))
