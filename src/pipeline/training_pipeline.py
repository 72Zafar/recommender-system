import sys
from src.exception import CustomException
from src.logger import logging

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.entity.config_entity import (
    DataIngestionConfig,
    DataTransformationConfig,
    Hybird_function_Config,
)
from src.entity.artifact_entity import (
    DataIngestionArtifact,
    DataTransformationArtifact,
    TFIDFVectorizerArtifact,
    SemanticEmbeddingArtifact,
    Hybird_function_Artifact,
)
from src.components.tfidf import TfidfRecommende
from src.components.Semantic_E_Recomm import SemanticEmbeddingRecommender
from src.components.hybird_function import HybirdRecommender


class TrainingPipeline:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()
        self.data_transformation_config = DataTransformationConfig()
        self.hybird_function_config = Hybird_function_Config()


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

    def start_tfidf_vectorizer(self, data_transformation_artifact: DataTransformationArtifact) -> TFIDFVectorizerArtifact:
        try:
            logging.info("Starting TF-IDF vectorization")
            tfidf_vectorizer = TfidfRecommende()
            return tfidf_vectorizer.initiate_tfidf_vectorization(data_transformation_artifact=data_transformation_artifact)
        except Exception as e:
            raise CustomException(e, sys)

    def start_semantic_embedding(self, data_transformation_artifact: DataTransformationArtifact) -> SemanticEmbeddingArtifact:
        try:
            logging.info("Starting semantic embedding")
            semantic_embedding = SemanticEmbeddingRecommender()
            return semantic_embedding.initiate_semantic_embedding(
                data_transformation_artifact=data_transformation_artifact
            )
        except Exception as e:
            raise CustomException(e, sys)

    def start_hybird_recommender(
        self,
        data_transformation_artifact: DataTransformationArtifact,
        tfidf_vectorizer_artifact: TFIDFVectorizerArtifact,
        semantic_embedding_artifact: SemanticEmbeddingArtifact,
    ) -> Hybird_function_Artifact:
        try:
            logging.info("Starting hybrid recommender")
            hybrid_recommender = HybirdRecommender(
                config=self.hybird_function_config,
                tfidf_artifact=tfidf_vectorizer_artifact,
                semantic_artifact=semantic_embedding_artifact,
            )
            logging.info(
                "Hybrid recommender initialized for transformed data at %s",
                data_transformation_artifact.transformed_data_file_path,
            )
            return hybrid_recommender.save()
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
            tfidf_vectorizer_artifact = self.start_tfidf_vectorizer(data_transformation_artifact=data_transformation_artifact)
            logging.info(f"TF-IDF vectorizer artifact: {tfidf_vectorizer_artifact}")
            semantic_embedding_artifact = self.start_semantic_embedding(data_transformation_artifact=data_transformation_artifact)
            logging.info(f"Semantic embedding artifact: {semantic_embedding_artifact}")
            hybird_function_artifact = self.start_hybird_recommender(
                data_transformation_artifact=data_transformation_artifact,
                tfidf_vectorizer_artifact=tfidf_vectorizer_artifact,
                semantic_embedding_artifact=semantic_embedding_artifact,
            )
            logging.info(f"Hybrid recommender artifact: {hybird_function_artifact}")
            return hybird_function_artifact
        except Exception as e:
            raise CustomException(e, sys)
    
