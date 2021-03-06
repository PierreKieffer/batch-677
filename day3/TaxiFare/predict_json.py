#!/usr/bin/env python3
import json
import sys
from google.cloud import storage
import os
import joblib
import pandas as pd 
from datetime import datetime

"""
HOW TO RUN : 

./predict_json.py '{"dropoff_latitude":40.74383544921875,"dropoff_longitude":-73.98143005371094,"key":"2015-01-27 13:08:24.0000002","passenger_count":1,"pickup_datetime":"2015-01-27 13:08:24 UTC","pickup_latitude":40.7638053894043,"pickup_longitude":-73.97332000732422}'
"""

os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="path/to/cred.json file"


def download_model(bucket="ml-bucket-pierrekieffer"):
    log("info", "download model from {}".format(bucket))

    client = storage.Client().bucket(bucket)

    storage_location = "models/simpletaxifare/model.joblib"
    blob = client.blob(storage_location)
    blob.download_to_filename('model.joblib')
    model = joblib.load('model.joblib')

    return model

def predict(input_json_data): 

    log("info", "run prediction on {}".format(input_json_data))

    BUCKET_NAME="ml-bucket-pierrekieffer"

    '''
    Download model
    '''
    model = download_model(BUCKET_NAME)

    '''
    Process input payload
    '''
    input_data = json.loads(input_json_data)
    input_df = pd.DataFrame.from_dict([input_data], orient = "columns")[["key", "pickup_datetime", "pickup_longitude", "pickup_latitude", "dropoff_longitude", "dropoff_latitude", "passenger_count"]]

    '''
    Prediction 

    '''
    pred = model.predict(input_df)
    return pred 

def log(level, message):

    '''
    Simple log display method.
    It is better to use a logger
    '''

    level = level.upper()
    now = datetime.now()
    log_date = now.strftime("%d/%m/%Y %H:%M:%S")

    log = "{} : {} : {}".format(log_date, level, message)
    print(log)


if __name__=="__main__":

    pred = predict(sys.argv[1])
    response = {"cost" : pred[0]}
    print(response)
