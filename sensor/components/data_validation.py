from sensor.entity import config_entity
from sensor.entity import artifact_entity
from sensor.logger import logging
from sensor.exception import SensorException
from typing import Optional
from scipy.stats import ks_2samp
from sensor import utils
import pandas as pd 
import numpy as np
import os,sys
from sensor.config import TARGET_COLUMN

class DataValidation:
    def __init__(self, 
                data_validation_config:config_entity.DataValidationConfig,
                data_ingestion_artifact:artifact_entity.DataIngestionArtifact):
                try:
                    logging.info(f"{'<<'*20}Data Validation {'>>*20'}")
                    self.data_validation_config = data_validation_config
                    self.data_ingestion_artifact = data_ingestion_artifact
                    self.validation_error = dict()
                except Exception as e:
                    raise SensorException(e, sys)

    def drop_missing_values_columns(self, df:pd.DataFrame , report_key_name:str)->Optional[pd.DataFrame]:
        """this method will return dataframe with deleted columns with missing values
        more than threshold"""
        try:
            thresold = self.data_validation_config.missing_thresold
            null_cols = df.isna().sum()/df.shape[0]
            logging.info(f"Selecting columns that contains missing values greater than {thresold}")
            drop_col_names = null_cols[null_cols>thresold].index
            logging.info(f"Dropping columns are {list(drop_col_names)}")
            self.validation_error[report_key_name] = list(drop_col_names)
            df.drop(list(drop_col_names) , inplace = True , axis=1)
            
            #Return null if no columns are left
            if len(df.columns)==0:
                return None
            return df

        except Exception as e:
            raise SensorException(e, sys)
        
    def is_required_columns_exists(self, base_df: pd.DataFrame , current_df: pd.DataFrame , report_key_name:str):
        try:
            base_cols = base_df.columns
            current_cols = current_df.columns

            missing_col = []
            for base_col in base_cols:    
                if base_col not in current_cols:
                    logging.info(f"Columns are missing {base_col}")
                    missing_col.append(base_col)
                
            if len(missing_col)>0:
                self.validation_error[report_key_name] = missing_col
                return False
            return True
        
        except Exception as e:
            raise SensorException(e, sys)

    def data_drift(self , base_df: pd.DataFrame , current_df: pd.DataFrame, report_key_name:str):
        try:
            drift_report = dict()
            base_cols = base_df.columns
            current_cols = current_df.columns
            for base_col in base_cols:
                base_data , current_data = base_df[base_col] , current_df[base_col]
                #Null hypothesis is that both column data drawn from same distrubtion
                logging.info(f"Hypothesis {base_cols}: {base_data.dtype}, {current_data.dtype} ")
                sample_distribution = ks_2samp(base_data, current_data)

                if sample_distribution.pvalue>0.05:
                    drift_report[base_col] = {
                        "pvalue": float(sample_distribution.pvalue),
                        "same distribution": True
                    }
                else:
                    drift_report[base_col] = {
                        "pvalue": float(sample_distribution.pvalue),
                        "same distribution": False
                    }

            self.validation_error[report_key_name] = drift_report


        except Exception as e:
            raise SensorException(e, sys)

    def initiate_data_validation(self):
        try:
            logging.info("Reading base data")
            base_df = pd.read_csv(self.data_validation_config.base_file_path)
            #Replacing na with np.NAN
            logging.info("replacing 'na' with np.NAN")
            base_df.replace({"na":np.NAN}, inplace = True)
            logging.info("Dropping missing values columns from base dataframe")
            base_df = self.drop_missing_values_columns(df=base_df , report_key_name="missing_values_from_base_dataframe")
            logging.info("Reading Train dataset")
            train_df = pd.read_csv(self.data_ingestion_artifact.train_file_path)
            logging.info("Reading Test Dataset")
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)
            
            #Dropping missing values from train and test data
            logging.info("Dropping missing values from train data")
            train_df = self.drop_missing_values_columns(df =train_df , report_key_name = "missing_values_in_train_data")
            logging.info("Dropping missing values from test data")
            test_df = self.drop_missing_values_columns(df =test_df, report_key_name = "missing_values_from_test_data")
            

            exclude_cols = [TARGET_COLUMN]
            base_df = utils.convert_cols_to_float(df = base_df, exclude_cols = exclude_cols)
            train_df = utils.convert_cols_to_float(df = train_df, exclude_cols = exclude_cols)
            test_df = utils.convert_cols_to_float(df = test_df, exclude_cols = exclude_cols)
            
            #Checking required columns exist 
            logging.info("Checking for required train columns")
            train_df_column_status = self.is_required_columns_exists(base_df = base_df, current_df = train_df, report_key_name = "missing_columns_within_train_dataset")
            logging.info("Checking for required test columns")
            test_df_column_status = self.is_required_columns_exists(base_df =base_df, current_df = test_df  , report_key_name = "missing_columns_wihtin_test_dataset")
        

            if train_df_column_status:
                logging.info("All the columns are available")
                self.data_drift(base_df = base_df , current_df = train_df , report_key_name = "data_drift_within_train_dataset")

            if test_df_column_status:
                logging.info("All the columns are available")
                self.data_drift(base_df = base_df , current_df = test_df , report_key_name = "data_drift_within_test_dataset")
            
             #Writing report
            logging.info("Writing report in yaml file")
            utils.write_yaml_file(file_path= self.data_validation_config.report_file_path, data=self.validation_error)
            data_validation_artifact = artifact_entity.DataValidationArtifact(report_file_path=self.data_validation_config.report_file_path)
            logging.info(f"Data validation artifact: {data_validation_artifact}")
            return data_validation_artifact

        except Exception as e:
            raise SensorException(e, sys)

