import os
import sys

# sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# print(sys.path)

from clearml import PipelineDecorator
from pipeline_steps import DataExtraction, DataPreprocessing, TrainModel


# import data_extraction as extraction
# import data_preprocessing as preprocessing
# import train_ml_model as train_model


@PipelineDecorator.component(return_values=["dataset_id"], cache=False)
def extract_data_from_mongodb(database: str, collection: str, id: str, connection_string: str) -> str:
    print("Running DataExtraction")
    dataset_id = DataExtraction(database, collection, id, connection_string).extract_data_from_db()
    # dataset_id = extraction.extract_data_from_db(database, collection, id)
    return dataset_id


@PipelineDecorator.component(return_values=["dataset_id"], cache=True)
def process_raw_data(dataset_id_input: str) -> str:
    dataset_id = DataPreprocessing(dataset_id_input).manage_preprocessing()
    # dataset_id = preprocessing.manage_preprocessing(dataset_id_input)
    return dataset_id


@PipelineDecorator.component(return_values=["model_id"], cache=True)
def train_model(dataset_id: str) -> str:
    model_id = TrainModel(dataset_id).manage_training()
    # model_id = train_model.train_model(dataset_id)
    return model_id


@PipelineDecorator.pipeline(name="MongoDB Pipline", project="Thesis", version="0.2.0")
def run_pipeline(database: str, collection: str, id: str, connection_string: str):
    print("Running Pipeline")
    dataset_id = extract_data_from_mongodb(database, collection, id, connection_string)
    dataset_id = process_raw_data(dataset_id)
    model_id = train_model(dataset_id)


if __name__ == "__main__":
    PipelineDecorator.set_default_execution_queue("services")
    PipelineDecorator.run_locally()
    # local connection string: mongodb://leon:thesis2024!@localhost:27017/
    run_pipeline(database="PQM", collection="all_V2", id="production_line_1",
                 connection_string="mongodb+srv://leon:thesis2024!@thesis.4kmqr.mongodb.net/?retryWrites=true&w=majority&appName=Thesis")
