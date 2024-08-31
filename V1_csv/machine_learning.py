from clearml import Task, Dataset
import pandas as pd
import os
from pathlib import Path
import seaborn as sns
from xgboost import XGBClassifier, plot_tree, plot_importance
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt

task = Task.init(project_name="Thesis", task_name="Train ML Model")

data_dir = Path.cwd().joinpath("data")

if not os.path.exists(data_dir.joinpath("d0c0254fd1854a29870ea6201a57fd83")):
    dataset = Dataset.get(
        dataset_id="d0c0254fd1854a29870ea6201a57fd83"
    ).get_mutable_local_copy(
        target_folder=data_dir.joinpath("d0c0254fd1854a29870ea6201a57fd83")
    )

df = pd.read_csv(data_dir.joinpath("d0c0254fd1854a29870ea6201a57fd83", "Stations_Merged_Encoded.csv"),
                 index_col="SerialNr")

task.upload_artifact(name="Stations_Merged_Encoded_DF", artifact_object=df)

X = df[["col_11", "col_12", "col_13", "col_21", "col_22", "col_23"]]
y = df["Error_Label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)

parameters = {
    "learning_rate": 0.05,
    "n_estimators": 600,
    "max_depth": 3,
    "seed": 42
}
task.connect(parameters)

model = XGBClassifier(**parameters)
model.fit(X_train, y_train)
predictions = model.predict(X_test)

task.get_logger().report_single_value(name="Accuracy", value=accuracy_score(y_test, predictions))
task.get_logger().report_single_value(name="MSE", value=mean_squared_error(y_test, predictions))

cm = confusion_matrix(y_test, predictions)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, cmap='Blues', fmt='g', xticklabels=df.Error.unique(),
            yticklabels=df.Error.unique())
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

plot_importance(model)
plt.show()
