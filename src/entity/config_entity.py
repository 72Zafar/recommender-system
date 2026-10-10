import os
from src.constants import *
from dataclasses import dataclass
from datetime import datetime

TIMESTAMP: str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

@dataclass
class TraingPipelineConfig:
    pipeline_name: str = PIPELINE_NAME
    artifacts_dir: str = os.path.join(ARTIFACTS_DIR, TIMESTAMP)
    timestamp: str = TIMESTAMP

training_pipeline_config: TraingPipelineConfig = TraingPipelineConfig()


class DataIngestionConfig:
    data_ingestion_dir: str = os.path.join(training_pipeline_config.artifacts_dir, DATA_INGESTION_DIR_NAME)
    collection_name: str = DATA_INGESTION_COLLECTION_NAME


class DataTransformationConfig:
    data_transformation_dir: str = os.path.join(training_pipeline_config.artifacts_dir, DATA_TRANSFORMATION_DIR_NAME)
    transformed_data_dir: str = os.path.join(data_transformation_dir, DATA_TRANSFORMATION_TRANSFORMED_DIR)