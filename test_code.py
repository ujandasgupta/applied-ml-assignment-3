# import unittest
# import joblib
# from score import score
# import requests
# import os

# class TestScoreFunction(unittest.TestCase):

#     @classmethod
#     def setUpClass(cls):
#         # Load a trained model for testing
#         cls.model = joblib.load("bestModelSGD.joblib")
    
#     def test_score_function(self):
#         # Smoke test
#         prediction, propensity = score("test text", self.model, 0.5)
#         self.assertIsNotNone(prediction)
#         self.assertIsNotNone(propensity)
        
#         # Format test
#         self.assertIsInstance(prediction, bool)
#         self.assertIsInstance(propensity, float)
        
#         # Prediction value check
#         self.assertIn(prediction, [True, False])
        
#         # Propensity score check
#         self.assertTrue(0 <= propensity <= 1)
        
#         # Threshold checks
#         _, propensity_zero = score("test text", self.model, 0)
#         self.assertTrue(propensity_zero > 0)
        
#         _, propensity_one = score("test text", self.model, 1)
#         self.assertTrue(propensity_one < 1)
        
#         # Test with obvious spam and non-spam inputs
#         spam_prediction, _ = score("Free money!!!", self.model, 0.5)
#         self.assertTrue(spam_prediction)
        
#         non_spam_prediction, _ = score("This is a regular email.", self.model, 0.5)
#         self.assertFalse(non_spam_prediction)

# class TestFlaskApp(unittest.TestCase):

#     def test_flask_app(self):
#         # Launch the flask app
#         os.system("flask run &")
        
#         # Give Flask a second to start
#         import time; time.sleep(3)
        
#         # Test the response from the localhost endpoint
#         response = requests.post("http://127.0.0.1:5000/score", json={"text": "test text"})
#         self.assertEqual(response.status_code, 200)
        
#         json_data = response.json()
#         self.assertIn('prediction', json_data)
#         self.assertIn('propensity', json_data)
        
#         # Close the flask app
#         os.system("kill $!")

# if __name__ == '__main__':
#     unittest.main()
import unittest
import os
import requests
from score import score
import joblib
import subprocess
import time
import pandas as pd
import json
import warnings

warnings.simplefilter("ignore")

class TestScoringFunction(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        return super().setUpClass()
    #     # Load the trained model for testing
    #     cls.loaded_model = joblib.load('sgd_classifier_model.joblib')
        
    #     # Load input texts from the CSV file
    #     cls.test_df = pd.read_csv("test.csv")
    def setUp(self):
        self.model = joblib.load('xgboost_model.pkl')
        self.vectorizer = joblib.load('tfidf_vectorizer.pkl')

    def test_score(self):
        text = "example spam text"
        threshold = 0.5
        prediction, propensity = score(text, self.model, threshold)
        self.assertIn(prediction, [True, False])
        self.assertTrue(0 <= propensity <= 1)

    @classmethod
    def tearDownClass(cls) -> None:
        return super().tearDownClass()

    # def test_score(self):
    #     # Load model and vectorizer for testing
    #     model = joblib.load('xgboost_model.joblib')
    #     # Example tests
    #     prediction, propensity = score('example spam text', model, 0.5)
    #     self.assertTrue(isinstance(prediction, bool))
    #     self.assertTrue(0 <= propensity <= 1)

class TestFlaskApp(unittest.TestCase):

    # @classmethod
    # def setUpClass(cls) -> None:
    #     cls.flask_process = subprocess.Popen(["python", "app.py"])
    #     time.sleep(10)  # allow for the flask server to start
    #     cls.test_dataset = pd.read_csv("test.csv")

    def test_flask(self):
        # Start the Flask app
        flask_process = subprocess.Popen(["python", "app.py"])
        time.sleep(10)
        # Test the /score endpoint
        response = requests.post(
            'http://127.0.0.1:5050/score', 
            data=json.dumps({'text': 'example spam text'}), 
            headers={"Content-Type": "application/json"}
        )
        self.assertEqual(response.status_code, 200)  # Check if the request was successful
        data = response.json()
        self.assertIn('prediction', data)
        self.assertIn('propensity', data)
        # Stop the Flask app
        flask_process.terminate()

    # @classmethod
    # def tearDownClass(cls):
    #     # Close Flask app using command line
    #     cls.flask_process.terminate()

if __name__ == '__main__':
    unittest.main()
