import joblib
import os
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd 
from xgboost import XGBRegressor


curr_path = os.path.dirname(os.path.realpath(__file__))

feat_cols = ['Distance','Haversine', 'Phour','Pmin','Dhour','Dmin','Temp','Humid','Solar','Dust']

xgb_final = joblib.load(curr_path + "/best_model.joblib")

print(xgb_final)

def predict_duration(attributes: np.ndarray):

    pred = xgb_final.predict(attributes)

    print("Duration Predicted")

    return int(pred[0])