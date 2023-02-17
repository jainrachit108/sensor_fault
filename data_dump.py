import pymongo
import pandas as pd
import json

# Provide the mongodb localhost url to connect python to mongodb.
mongo_client = pymongo.MongoClient('mongodb://localhost:27017')
DATABASE_NAME = 'aps'
COLLECTION_NAME = 'sensor'

DATA_PATH = "C:/Users/91888/OneDrive/Desktop/sensor_fault/aps_failure_training_set1.csv"

if __name__ == '__main__':
    df =pd.read_csv(DATA_PATH)
    print(f'Rows and columns are {df.shape}')
    df.reset_index(drop = True, inplace = True)
    json_record = list(json.loads(df.T.to_json()).values())
    print(json_record[0])

    mongo_client[DATABASE_NAME][COLLECTION_NAME].insert_many(json_record)