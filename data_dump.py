import pymongo
import pandas as pd
import json
from sensor.config import mongo_client
# Provide the mongodb localhost url to connect python to mongodb.

DATABASE_NAME = 'aps'
COLLECTION_NAME = 'sensor'

DATA_PATH = "aps_failure_training_set1.csv"

if __name__ == '__main__':
    df =pd.read_csv(DATA_PATH)
    print(f'Rows and columns are {df.shape}')
    df.reset_index(drop = True, inplace = True)
    json_record = list(json.loads(df.T.to_json()).values())
    print(json_record[0])

    mongo_client[DATABASE_NAME][COLLECTION_NAME].insert_many(json_record)