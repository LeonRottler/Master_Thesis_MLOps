from clearml import Task, Dataset
import pandas as pd
import matplotlib.pyplot as plt
import os

task = Task.init(project_name="Thesis", task_name="Data Visualization")

dataset = Dataset.get(dataset_project="Thesis", dataset_name="StationsMerged_Transformed.csv").get_local_copy()
df = pd.read_csv(os.path.join(dataset, "StationsMerged.csv"), index_col="SerialNr")

task.upload_artifact(name="UsedDataset", artifact_object=df)

error_categories = df.Error.unique()

for error_category in error_categories:
    plt.scatter(df.index[df["Error"] == error_category],
                df.loc[df["Error"] == error_category, "col_11"],
                label=error_category)
plt.xlabel("Serial Number")
plt.ylabel("col_11")
plt.ylim(65, 110)
plt.title("Correlation between col_11 and Error")
plt.legend(loc="upper left")
plt.show()

for error_category in error_categories:
    plt.scatter(df.index[df["Error"] == "Angel_to_low"],
                df.loc[df["Error"] == "Angel_to_low", "col_11"])
plt.xlabel("Serial Number")
plt.ylabel("col_11")
plt.ylim(65, 110)
plt.title("Correlation between col_11 and Angel to low Error")
# plt.legend(loc="upper left")
plt.show()

for error_category in error_categories:
    plt.scatter(df.index[df["Error"] == error_category],
                df.loc[df["Error"] == error_category, "col_12"],
                label=error_category)
plt.xlabel("Serial Number")
plt.ylabel("col_12")
plt.title("Correlation between col_12 and Error")
plt.legend(loc="upper left")
plt.show()
