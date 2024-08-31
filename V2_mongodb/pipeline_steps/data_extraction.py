import os
from pymongo import MongoClient
from clearml import Task, Dataset
import json
from bson import ObjectId


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
        print("running method extract_data_from_db")
        self.save_config(self.database, self.collection, self.id)
        data = self.get_mongo_db_data(self.database, self.collection, self.id)
        self.client.close()
        return self.create_clearml_dataset(data, id)


if __name__ == "__main__":
    extraction = DataExtraction("PQM", "all_V2", "production_line_1").extract_data_from_db()
