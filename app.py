from flask import Flask, request, jsonify
from score import *
import joblib
import os

app = Flask(__name__)
model = joblib.load('bestModelSGD.joblib')  

@app.route('/score', methods=['POST'])
def score_endpoint():
    data = request.get_json()
    text = data.get('text')
    threshold = data.get('threshold', 0.5) 
    prediction, propensity = score(text, model, threshold)
    return jsonify({'prediction': prediction, 'propensity': propensity})

if __name__ == "__main__":
    port = int(os.environ.get("FLASK_RUN_PORT", 5050))  # Default to 5000 if not specified
    app.run(debug=True, port=5050)

