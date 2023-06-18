import math

import joblib
from keras.models import load_model
from flask import Flask, request, jsonify
from flask_cors import CORS
from math import radians, degrees, sin, cos, asin, acos
import random
import pandas as pd


app = Flask(__name__)
#Set up Flask to bypass CORS:
cors = CORS(app)
ss = joblib.load('standard_scaler.joblib')
model = joblib.load('voting_classifier_model.joblib')
@app.route("/predict", methods=["POST"])
def GetPredictionApi():
  data = request.get_json()
  latitude = float(data['latitude'])
  longitude = float(data['longitude'])
  radius = float(data['radius'])


  final_df = pd.DataFrame([[
    float(data['cdi']),
    float(data['mmi']),
    float(data['sig']),
    float(data['net']),
    float(data['nst']),
    float(data['dmin']),
    float(data['gap']),
    float(data['magType']),
    float(data['depth']),
    float(data['latitude']),
    float(data['longitude']),
  ]], columns=['cdi', 'mmi', 'sig','net','nst','dmin','gap','magType','depth','latitude','longitude',])

  x_scaled = ss.transform(final_df)
  confidence_scores = model.predict_proba(x_scaled)

  return jsonify({'prediction':str(confidence_scores)})

if __name__ == "__main__":
 app.run(debug=True)