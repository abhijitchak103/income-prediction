import os, sys
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.base import TransformerMixin

import pandas as pd
import numpy as np

from dataclasses import dataclass


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')



class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()   


    def get_data_transformation_obj(self):
        try:
            logging.info('Data Transformation Started')

            numerical_columns = ['age',
                                 'children',
                                 'bmi']

            logging.info('Defining Numerical Pipeline')
            numerical_pipeline = Pipeline(
                steps = [
                ('Imputer', SimpleImputer(strategy='median')),
                ('Scaler', StandardScaler())
                ]
            )

            categorical_binary = ['smoker',
                                   'sex']
            
            categorical = ['region']

            logging.info('Defining Categorical Pipeline')
            categorical_binary_pipeline = Pipeline(
                steps = [
                ('Imputer', SimpleImputer(strategy='most_frequent')),
                ('Binary_Encoder', OneHotEncoder())
                ]
            )

            categorical_pipeline = Pipeline(
                steps= [
                    ('Imputer', SimpleImputer(strategy='most_frequent')),
                    ('Encoder', OneHotEncoder())
                ]
            )

            logging.info('Defining Preprocessor')
            preprocessor = ColumnTransformer(
                [
                ('Numerical_Pipeline', numerical_pipeline, numerical_columns),
                ('Categorical_Binary_Pipeline', categorical_binary_pipeline, categorical_binary),
                ('Categorical_Pipeline', categorical_pipeline, categorical)
                ]
            )

            logging.info('Pipeline Created')

            return preprocessor
        
        except Exception as e:
            logging.info('Error Occured in get_data_transformation_obj')
            raise CustomException(e, sys)


    def initiate_data_transformation(self, train_path, test_path):
        try:
            logging.info('Reading Training and test Data')
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info('Read Train and test data')

            logging.info(f"Training dataset head: \n{train_df.head().to_string()}")
            logging.info(f"Test dataset head: \n{test_df.head().to_string()}")
            
            logging.info('Obtaining Preprocessor Object')
            preprocessor_obj = self.get_data_transformation_obj()

            target_column = 'expenses'
            
            input_feature_train_df = train_df.drop(columns = target_column, axis = 1)
            logging.info(f"Training input feature dataset head: \n{input_feature_train_df.head().to_string()}")
            target_feature_train_df = train_df[target_column]

            input_feature_test_df = test_df.drop(columns = target_column, axis = 1)
            logging.info(f"Test input feature dataset head: \n{input_feature_test_df.head().to_string()}")
            target_feature_test_df = test_df[target_column]
            

            logging.info('Transforming using preprocessor object')
            input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor_obj.transform(input_feature_test_df)
            logging.info('Train and test data transformed')

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor_obj
            )

            logging.info('Preprocessor pickle file saved')

            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
        
        except Exception as e:
            logging.info('Error occured in initiate_data_transformation')
            raise CustomException(e, sys)