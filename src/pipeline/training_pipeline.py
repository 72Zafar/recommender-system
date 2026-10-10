import sys
from src.exception import CustomException
from src.logger import logging

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.entity.config_entity import  DataIngestionConfig,DataTransformationConfig
from src.entity.artifact_entity import DataIngestionArtifact,DataTransformationArtifact

class TrainingPipeline:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()
        self.data_transformation_config = DataTransformationConfig()

    def start_data_ingestion(self) -> DataIngestionArtifact:
        try:
            logging.info("Starting data ingestion")
            data_ingestion = DataIngestion(data_ingestion_config=self.data_ingestion_config)
            return data_ingestion.initiate_data_ingestion()
        except Exception as e:
            raise CustomException(e, sys)

    def start_data_transformation(self, data_ingestion_artifact: DataIngestionArtifact) -> DataTransformationArtifact:
        try:
            logging.info("Starting data transformation")
            data_transformation = DataTransformation(data_ingestion_artifact=data_ingestion_artifact, data_transformation_config=self.data_transformation_config)
            return data_transformation.initiate_data_transformation()
        except Exception as e:
            raise CustomException(e, sys)
        

    def run_pipeline(self):
        """
        This function will run the entire training pipeline
        """
        try:
            data_ingestion_artifact = self.start_data_ingestion()
            logging.info(f"Data ingestion artifact: {data_ingestion_artifact}")
            data_transformation_artifact = self.start_data_transformation(data_ingestion_artifact=data_ingestion_artifact)
            logging.info(f"Data transformation artifact: {data_transformation_artifact}")
        except Exception as e:
            raise CustomException(e, sys)
    
