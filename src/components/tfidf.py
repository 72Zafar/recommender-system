from src.logger import logging
from src.exception import CustomException
from src.entity.artifact_entity import DataTransformationArtifact, TFIDFVectorizerArtifact
from src.entity.config_entity import TFIDFVectorizerConfig
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import os
import sys
import joblib
from scipy.sparse import save_npz


class TfidfRecommende:
    def __init__(self, max_features: int = 20000):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words="english",
            ngram_range=(1,2),
            min_df = 2,
            max_df = 0.95,
            sublinear_tf=True
        )
        self.matrix = None
        self.course_id_to_index: dict[str, int] = {}

    def fit(self, data: pd.DataFrame) -> "TfidfRecommende":
        try:
            logging.info("Fitting TfidfRecommender...")
            self.matrix = self.vectorizer.fit_transform(data["course_text"].astype(str))
            self.course_id_to_index = pd.Series(data.index, index=data["course_id"]).to_dict()
            return self
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_tfidf_vectorization(
        self,
        data_transformation_artifact: DataTransformationArtifact,
    ) -> TFIDFVectorizerArtifact:
        try:
            logging.info("Starting TF-IDF vectorization process")
            data = pd.read_csv(data_transformation_artifact.transformed_data_file_path)
            self.fit(data)

            config = TFIDFVectorizerConfig()
            self.save(config)

            return TFIDFVectorizerArtifact(
                tfidf_vectorizer_file_path=config.tfidf_vectorizer_file_path,
                tfidf_matrix_file_path=config.tfidf_matrix_file_path,
                course_id_to_index_file_path=config.course_id_to_index_file_path,
            )
        except Exception as e:
            raise CustomException(e, sys)

    def recommender(self,data : pd.DataFrame, course_id: str, top_n: int = 5)-> pd.DataFrame:
        try: 
            logging.info("Generating recommendations...")
            if course_id not in self.course_id_to_index:
                raise ValueError(f"Course ID {course_id} not found in the dataset.")
            index = self.course_id_to_index[course_id]
            score = cosine_similarity(self.matrix[index], self.matrix).flatten()
            indices = [item for item in score.argsort()[::-1] if item != index][:top_n]
            result = data.iloc[indices].copy()
            result["similarity_score"] = score[indices]
            return result.reset_index(drop=True)
        except Exception as e:
            raise CustomException(e, sys)


    def save(self, config: TFIDFVectorizerConfig) -> None:
        try:
            logging.info("Saving TfidfRecommender artifacts...")
            os.makedirs(config.tfidf_vectorizer_dir, exist_ok=True)
            joblib.dump(self.vectorizer, config.tfidf_vectorizer_file_path)
            save_npz(config.tfidf_matrix_file_path,self.matrix )
            joblib.dump(self.course_id_to_index, config.course_id_to_index_file_path)
            logging.info("TfidfRecommender artifacts saved successfully.")
        except Exception as e:
            raise CustomException(e, sys)


TfidfRecommender = TfidfRecommende
        