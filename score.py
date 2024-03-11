import joblib
import sklearn
import numpy as np
import pandas as pd
from typing import Tuple
import numpy as np
from sklearn.pipeline import Pipeline

def score(text, model_path, threshold):
    # Load the trained model pipeline
    model_pipeline = joblib.load('bestModelSGD.joblib')
    
    # Access the TF-IDF Vectorizer directly by its named step in the pipeline
    text_vectorized = model_pipeline.named_steps['tfidf'].transform([text])
    
    # Get the classifier step
    classifier = model_pipeline.named_steps['sgd'] 
    
     # Use the decision function to get the distance from the decision boundary
    decision_score = classifier.decision_function(text_vectorized)[0]
    
    # Apply the threshold to the decision score to get the binary prediction
    # Note: You might need to adjust the comparison based on your model's conventions
    prediction = decision_score > threshold
    
    # Normalize the decision score to a 0-1 scale as a proxy for propensity
    # This step is optional and a simplification; the exact mapping depends on your use case
    propensity = (decision_score - decision_score.min()) / (decision_score.max() - decision_score.min())
    
    return bool(prediction), propensity
