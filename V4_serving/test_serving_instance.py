import time
import pandas as pd
import json
from clearml import Dataset
from sklearn.model_selection import train_test_split
import requests

url = "http://localhost:8080/serve/thesis_test"

data_path = Dataset.get(dataset_name="processed_dataset").get_local_copy()
merged_df = pd.read_csv(f"{data_path}/merged_input_data.csv", index_col="SerialNr")

X = merged_df.filter(regex="col_")
y = merged_df["Error_Label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)

calls = 100

for call_to_api in range(1, calls):
    dict_payload = X[:calls].to_dict(orient="records")[call_to_api]
    print(dict_payload)
    response = requests.post(url, json=dict_payload)
    print(response.json())
    time.sleep(1)
