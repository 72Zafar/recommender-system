import sys
from src.exception import CustomException
from src.logger import logging

from src.components.data_ingestion import DataIngestion
from src.entity.config_entity import  DataIngestionConfig
from src.entity.artifact_entity import DataIngestionArtifact

class TrainingPipeline:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()

    def start_data_ingestion(self) -> DataIngestionArtifact:
        try:
            logging.info("Starting data ingestion")
            data_ingestion = DataIngestion(data_ingestion_config=self.data_ingestion_config)
            return data_ingestion.initiate_data_ingestion()
        except Exception as e:
            raise CustomException(e, sys)

    def run_pipeline(self):
        """
        This function will run the entire training pipeline
        """
        try:
            data_ingestion_artifact = self.start_data_ingestion()
            logging.info(f"Data ingestion artifact: {data_ingestion_artifact}")
        except Exception as e:
            raise CustomException(e, sys)
    
