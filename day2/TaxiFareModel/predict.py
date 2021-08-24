#!/usr/bin/env python3
import json 
import joblib 
import pandas as pd  
import sys


def predict_client(input_json) : 

    model = joblib.load("model.joblib")

    df = pd.DataFrame.from_dict([json.loads(input_json)], orient = "columns")[["key", "pickup_datetime", "pickup_longitude", "pickup_latitude", "dropoff_longitude", "dropoff_latitude", "passenger_count"]]

    pred = model.predict(df)
    return pred[0]

if __name__=="__main__":
    prediction = predict_client(sys.argv[1])
    print("taxi fare predictiion = {} $".format(prediction))

