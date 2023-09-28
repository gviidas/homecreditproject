from flask import Flask, request, jsonify
import joblib
import pandas as pd
import json
import os
from datetime import datetime

app = Flask(__name__)

# Load the model when the Flask app starts
lgbm_model = joblib.load('lgbm_undersampled.pkl')

@app.route('/predict/target', methods=['POST'])
def predict_target():
    start_time = datetime.now()

    data = request.json
    df = pd.DataFrame(data, index=[0])

    predictions = lgbm_model.predict(df).tolist()

    target_probabilities = lgbm_model.predict_proba(df).tolist()[0]

    end_time = datetime.now()
    processing_time = (end_time - start_time).total_seconds()

    return jsonify({
        "target": predictions,
        "target_probability": target_probabilities,
        "processing_time": processing_time
    })

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port)