from clearml import Task, Dataset
import pandas as pd
import os
import json
from pathlib import Path
from sklearn import preprocessing


class DataPreprocessing:
    def __init__(self, dataset_id: str):
        self.task = Task.current_task()
        # self.task.set_base_docker(docker_image="python:3.12")
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


if __name__ == "__main__":
    updated_dataset_id = DataPreprocessing("252849eabfeb47bdb6593310fdde8a74").manage_preprocessing()
    # updated_dataset_id = manage_preprocessing("252849eabfeb47bdb6593310fdde8a74")
    print(updated_dataset_id)
