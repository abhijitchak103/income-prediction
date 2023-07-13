import os, sys
import pandas as pd
import numpy as np

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_model

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR


from dataclasses import dataclass


@dataclass
class ModelTrainerConfig:
    model_path = os.path.join('artifacts', 'model.pkl')


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()


    def initiate_model_trainer(self, train_arr, test_arr):
        try:
            logging.info('Getting X_train, X_test, y_train, y_test')
            X_train, y_train, X_test, y_test = (
                train_arr.drop(columns = 'expenses', axis=1),
                train_arr.expenses,
                test_arr.drop(columns = 'expenses', axis=1),
                test_arr.expenses
            )

            models = {
                'Random Forest Regressor': RandomForestRegressor(
                            n_estimators=360,
                            min_samples_leaf=4,
                            min_samples_split=10,
                            max_features='auto',
                            max_depth=6
                            )
            }

            model_report = evaluate_model(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test, models=models)
            print(model_report)
            print("="*40)
            logging.info(f'Model Report: {model_report}')

            logging.info('Fetching best model')
            best_model_score = max(model_report.values())

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            print(f"Best Model Found:\n\tModel Name: {best_model_name},\n\tR2 Score: {best_model_score}")
            print("="*40)
            logging.info(f"Best Model Found:\n\tModel Name: {best_model_name},\n\tR2 Score: {best_model_score}")

            save_object(
                file_path=self.model_trainer_config.model_path,
                obj=best_model
            )

        except Exception as e:
            logging.info('Error Occured in ModelTrainer.initiate_model_trainer')
            raise CustomException(e, sys)
