from pathlib import Path
import pandas as pd
from pymongo import MongoClient

csv_fileA = Path.cwd().joinpath("data", "v2", "StationA.csv")
dataA = pd.read_csv(csv_fileA)
data_dictA = dataA.to_dict("records")

csv_fileB = Path.cwd().joinpath("data", "v2", "StationB.csv")
dataB = pd.read_csv(csv_fileB)
data_dictB = dataB.to_dict("records")

csv_fileC = Path.cwd().joinpath("data", "v2", "StationC.csv")
dataC = pd.read_csv(csv_fileC)
data_dictC = dataC.to_dict("records")

mongoClient = MongoClient(
    "mongodb://leon:thesis2024!@localhost:27017/")
db = mongoClient["PQM"]
collection = db["thesis"]

document_stations = {
    "id": "production_line_1",
    "test_stations": {
        "StationA": data_dictA,
        "StationB": data_dictB,
        "StationC": data_dictC
    }
}
collection.insert_one(document_stations)

mongoClient.close()
