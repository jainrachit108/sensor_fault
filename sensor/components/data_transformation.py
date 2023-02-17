import os,sys
from sensor.entity import config_entity
from sensor.entity import artifact_entity
from sensor.exception import SensorException
from sensor.logger import logging
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import LabelEncoder
from imblearn.combine import SMOTETomek
from sensor.config import TARGET_COLUMN
from sklearn.pipeline import Pipeline
from sensor import utils


class DataTransformation:
    def __init__(self, data_ingestion_artifact:artifact_entity.DataIngestionArtifact , 
                    data_transformation_config:config_entity.DataTransformationConfig):
        try:
            logging.info(f"{'>>'*20}Data Transfomation{'<<'*20}")
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_transformation_config = data_transformation_config

        except Exception as e:
            raise SensorException(e, sys)


    @classmethod
    def get_data_transformer_object(cls):
        try:
            simple_imputer = SimpleImputer(strategy = 'constant' , fill_value = 0)      
            robust_scaler = RobustScaler()
            pipeline = Pipeline(steps = [
                ('Imputer', simple_imputer),
                ('Scaler', robust_scaler)
            ])
            return pipeline
        except Exception as e:
            raise SensorException(e, sys)


    def initiate_data_transformation(self ,)->artifact_entity.DataTransformationArtifact:
        try:
            logging.info("Initiating Data transformation")
            #Selecting training and testing files
            train_df = pd.read_csv(self.data_ingestion_artifact.train_file_path)
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)

            #Selelcting input feature from train and test data
            input_feature_train_data = train_df.drop(TARGET_COLUMN , axis =1)
            input_feature_test_data = test_df.drop(TARGET_COLUMN , axis =1)

            #Selecting target feature from train and test dataframe
            target_feature_train_df = train_df[TARGET_COLUMN]
            target_feature_test_df = test_df[TARGET_COLUMN] 

            
            label_encoder = LabelEncoder()
            label_encoder.fit(target_feature_train_df)

            #transformation on target columns 
            target_feature_train_arr = label_encoder.transform(target_feature_train_df)
            target_feature_test_arr = label_encoder.transform(target_feature_test_df)

            transformation_pipeline = DataTransformation.get_data_transformer_object()
            transformation_pipeline.fit(input_feature_train_data)

            input_feature_train_arr = transformation_pipeline.transform(input_feature_train_data)
            input_feature_test_arr = transformation_pipeline.transform(input_feature_test_data)

            smt = SMOTETomek(sampling_strategy = "minority")
            logging.info(f"Before resampling the training sample size is {input_feature_train_arr.shape} and target feature size is {target_feature_train_arr.shape}")
            input_feature_train_arr , target_feature_train_arr = smt.fit_resample(input_feature_train_arr , target_feature_train_arr)
            logging.info(f"After resampling the training sample size is {input_feature_train_arr.shape} and target feature size is {target_feature_train_arr.shape}")

            logging.info(f"Before resampling the test sample size is {input_feature_test_arr.shape} and target feature size is {target_feature_test_arr.shape}")
            input_feature_test_arr , target_feature_test_arr = smt.fit_resample(input_feature_test_arr , target_feature_test_arr)
            logging.info(f"After resampling the test sample size is {input_feature_test_arr.shape} and target feature size is {target_feature_test_arr.shape}")

            #Target_encoder
            train_arr = np.c_[input_feature_train_arr , target_feature_train_arr]
            test_arr = np.c_[input_feature_test_arr , target_feature_test_arr]

            #Save numpy array

            utils.save_numpy_array_data(file_path = self.data_transformation_config.transformed_train_path ,array= train_arr)

            utils.save_numpy_array_data(file_path = self.data_transformation_config.transformed_test_path ,array = test_arr)

            utils.save_object(filepath = self.data_transformation_config.transform_object_path , obj = transformation_pipeline)

            utils.save_object(filepath = self.data_transformation_config.target_encoder_path , obj = label_encoder)


            data_transformation_artifact = artifact_entity.DataTransformationArtifact(
                transform_object_path = self.data_transformation_config.transform_object_path,
                transformed_train_path = self.data_transformation_config.transformed_train_path, 
                transformed_test_path = self.data_transformation_config.transformed_test_path,
                target_encoder_path = self.data_transformation_config.target_encoder_path
            )

            logging.info(f"Data transformation object {data_transformation_artifact}")
            return data_transformation_artifact

        except Exception as e:
            raise SensorException(e , sys)
