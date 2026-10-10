import os
import sys
from pandas import DataFrame

import pandas as pd

from src.constants import DATA_INGESTION_INGESTED_DIR, FILE_NAME
from src.entity.artifact_entity import DataIngestionArtifact
from src.entity.config_entity import DataIngestionConfig
from src.logger import logging
from src.exception import CustomException


class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig):
        try:
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise CustomException(e,sys)

    def load_data(self, file_path: str) -> DataFrame:
        try:
            logging.info(f"Loading data from {file_path}")
            return pd.read_csv(file_path)
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        try:
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
            source_file_path = os.path.join(project_root, "Data", FILE_NAME)
            ingested_dir = os.path.join(
                self.data_ingestion_config.data_ingestion_dir,
                DATA_INGESTION_INGESTED_DIR,
            )
            ingested_file_path = os.path.join(ingested_dir, FILE_NAME)

            logging.info(f"Starting data ingestion from {source_file_path}")
            data = self.load_data(source_file_path)
            os.makedirs(ingested_dir, exist_ok=True)
            data.to_csv(ingested_file_path, index=False)
            logging.info(f"Saved ingested data to {ingested_file_path}")

            return DataIngestionArtifact(data_file_path=ingested_file_path)
        except Exception as e:
            raise CustomException(e, sys)

        