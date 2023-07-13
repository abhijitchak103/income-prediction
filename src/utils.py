import os, sys, pickle

from src.logger import logging
from src.exception import CustomException

import pandas as pd
import numpy as np
from sklearn.metrics import r2_score


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, 'wb') as path:
            pickle.dump(obj, path)

    except Exception as e:
        logging.info('Error occured in save_object')
        raise CustomException(e, sys)
    

def evaluate_model(X_train, X_test, y_train, y_test, models):
    try:
        report = {}
        for key, value in models.items():
            model = value

            logging.info(f'Training data with {value} model')
            model.fit(X_train, y_train)
            logging.info('Data trained')

            logging.info(f'Predicting with {value} model')
            y_pred = model.predict(X_test)
            logging.info('Prediction Complete')

            logging.info('Evaluating r2 scores for test data')
            test_model_score = r2_score(y_true=y_test, y_pred=y_pred)
            logging.info(f'Obtained R2 score for {value} model')

            report[key] = test_model_score
        
        return report
    except Exception as e:
        logging.info('Error occured in evaluate_model stage')
        raise CustomException(e, sys)
    

def load_object(file_path):
    try:
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        logging.info('Error occured in utils.load_object')
        raise CustomException(e, sys)