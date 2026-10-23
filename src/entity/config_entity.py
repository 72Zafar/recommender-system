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


class TFIDFVectorizerConfig:
    tfidf_vectorizer_dir: str = os.path.join(training_pipeline_config.artifacts_dir, TFIDF_DIR_NAME)
    tfidf_vectorizer_file_path: str = os.path.join(tfidf_vectorizer_dir, TFIDF_OBJECT_FILE_NAME)
    tfidf_matrix_file_path: str = os.path.join(tfidf_vectorizer_dir, TFIDF_MATRIX_FILE_NAME)
    course_id_to_index_file_path: str = os.path.join(tfidf_vectorizer_dir, TFIDF_CI_INDEX_FILE_NAME)


class SemanticEmbeddingConfig:
    embedding_dir: str = os.path.join(training_pipeline_config.artifacts_dir, EMBEDDING_DIR_NAME)
    embedding_file_path: str = os.path.join(embedding_dir, COURSE_EMBEDDING_FILE_NAME)
    model_name_file_path: str = os.path.join(embedding_dir, MODEL_NAME_TEXT)

class Hybird_function_Config:
    hybird_model_dir: str = os.path.join(training_pipeline_config.artifacts_dir, HYBIRD_DIR_NAME)
    hybird_model_file_path: str = os.path.join(hybird_model_dir, HYBIRD_MODEL_NAME)