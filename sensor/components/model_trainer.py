import os,sys
from sensor.entity import config_entity
from sensor.entity import artifact_entity
from sensor.exception import SensorException
from sensor.logger import logging
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sensor import utils
from sklearn.metrics import f1_score



class ModelTrainer:
    def __init__(self, model_trainer_config = config_entity.ModelTrainerConfig ,
                data_transformation_artifact = artifact_entity.DataTransformationArtifact):

        try:
            logging.info(f"{'>>'*20}Model Trainer {'<<'*20} ")
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact


        except Exception as e:
            raise SensorException(e, sys)


    def train_model(self, x ,y):
        try:
            xgb_clf = XGBClassifier()
            xgb_clf.fit(x,y)
            return xgb_clf
        
        except Exception as e:
            raise SensorException(e, sys)

    def initiate_model_trainer(self, )->artifact_entity.ModelTrainerArtifact:
        try:
            logging.info("Loading training and testing data for Model Training")
            train_arr = utils.load_numpy_array_data(file_path = self.data_transformation_artifact.transformed_train_path)
            test_arr = utils.load_numpy_array_data(file_path = self.data_transformation_artifact.transformed_test_path)

            logging.info("Splitting training and testing array from both train and test data")
            x_train , y_train = train_arr[:,:-1], train_arr[:,-1]
            x_test , y_test = test_arr[:,:-1] , test_arr[:,-1]

            logging.info("train the model")
            model = self.train_model(x = x_train, y = y_train)

            logging.info("Calculating f1 train score")
            yhat_train = model.predict(x_train)
            f1_train_score = f1_score(y_true = y_train , y_pred = yhat_train)
            
            logging.info("Calculating f1 test score")
            yhat_test = model.predict(x_test)
            f1_test_score = f1_score(y_true =y_test , y_pred = yhat_test)
            
            logging.info(f"train score {f1_train_score} and test score {f1_test_score}")
            #checking overfitting and underfitting
            logging.info("Checking if model is underfitted or not")
            if f1_test_score < self.model_trainer_config.expected_score:
                raise Exception(f"Model is not good as it is not giving expected accuracy, Expected accuracy = {self.model_train_config.expected_score} and Actual accuracy : {f1_test_score}")
            
            logging.info("Checking model is overfitted or not")
            diff = abs(f1_train_score-f1_test_score)
            if diff>self.model_trainer_config.overfitting_threshold:
                raise Exception(f"Train and test score diff: {diff} is more than overfitting threshold {self.model_trainer_config.overfitting_threshold}")

            #save the trained model
            logging.info(f"Saving mode object")
            utils.save_object(filepath=self.model_trainer_config.model_path, obj=model)



            logging.info("Preparing artifacts")
            model_trainer_artifact = artifact_entity.ModelTrainerArtifact(model_path = self.model_trainer_config.model_path ,
                    f1_train_score=f1_train_score , f1_test_score = f1_test_score )
            logging.info(f"Model trainer artifact: {model_trainer_artifact}")
            return model_trainer_artifact
            

        except Exception as e:
            raise SensorException(e, sys)