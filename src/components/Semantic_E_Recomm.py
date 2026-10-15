from pathlib import Path
from importlib import import_module
import os

import numpy as np
import pandas as pd

from src.entity.artifact_entity import DataTransformationArtifact, SemanticEmbeddingArtifact
from src.entity.config_entity import SemanticEmbeddingConfig
from src.constants import *
from src.exception import CustomException
from src.logger import logging
import sys


class SemanticEmbeddingRecommender:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        logging.info("Initializing SemanticEmbeddingRecommender...")
        self.model_name = model_name
        self.model = None
        self.embeddings: np.ndarray | None = None
        self.course_id_to_index: dict[str, int] = {}


    def fit(self, data:pd.DataFrame)-> "SemanticEmbeddingRecommender":
        try: 
            sentence_transformers = import_module("sentence_transformers")
        except ImportError as error:
            raise RuntimeError("Install sentence-transformers to train semantic embeddings.") from error
        self.model = sentence_transformers.SentenceTransformer(self.model_name)
        self.model.max_seq_length = 128
        self.embeddings = self.model.encode(
            data["course_text"].tolist(),
            batch_size = 32,
            show_progress_bar=True,
            normalize_embeddings=True,
        )
        self.course_id_to_index = pd.Series(data.index, index=data["course_id"]).to_dict()
        return self

    def initiate_semantic_embedding(
        self,
        data_transformation_artifact: DataTransformationArtifact,
    ) -> SemanticEmbeddingArtifact:
        try:
            logging.info("Starting semantic embedding process")
            data = pd.read_csv(data_transformation_artifact.transformed_data_file_path)
            self.fit(data)

            config = SemanticEmbeddingConfig()
            self.save(config)

            return SemanticEmbeddingArtifact(
                embedding_file_path=config.embedding_file_path,
                model_name_file_path=config.model_name_file_path,
            )
        except Exception as e:
            raise CustomException(e, sys)

    def semantic_recommender(self, data: pd.DataFrame, course_id: str, top_n: int = 5)->pd.DataFrame:
        try:
            logging.info("Generating semantic recommendations...")
            if course_id not in self.course_id_to_index:
                raise ValueError(f"Course ID {course_id} not found in the dataset.")
            index = self.course_id_to_index[course_id]
            scores = self.embeddings @ self.embeddings[index]
            indices = [item for item in scores.argsort()[::-1] if item != index][:top_n]
            result = data.iloc[indices].copy()
            result["similarity_score"] = scores[indices]
            return result.reset_index(drop=True)
        except Exception as e:  
            raise CustomException(e, sys)

    def save(self, config: SemanticEmbeddingConfig) -> None:
        logging.info("Saving SemanticEmbeddingRecommender artifacts...")
        if self.embeddings is None:
            raise ValueError("Fit the recommender before saving semantic embeddings.")
        os.makedirs(config.embedding_dir, exist_ok=True)
        np.save(config.embedding_file_path, self.embeddings)
        Path(config.model_name_file_path).write_text(self.model_name, encoding="utf-8")

    
        

