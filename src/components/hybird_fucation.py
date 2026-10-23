from src.logger import logging
from src.entity.config_entity import Hybird_function_Config
from src.entity.artifact_entity import TFIDFVectorizerArtifact,SemanticEmbeddingArtifact
from src.entity.artifact_entity import Hybird_function_Artifact
import joblib
import numpy as np
import os
import pandas as pd
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity


class HybirdRecommender:
    DEFAULT_WEIGHTS = {
        "semantic": 0.50,
        "tfidf": 0.20,
        "popularity": 0.20,
        "rating": 0.10,
    }

    def __init__(
        self,
        config: Hybird_function_Config,
        tfidf_artifact: TFIDFVectorizerArtifact,
        semantic_artifact: SemanticEmbeddingArtifact,
        weights: dict[str, float] | None = None,
    ):
        self.config = config
        self.tfidf_artifact = tfidf_artifact
        self.semantic_artifact = semantic_artifact
        self.weights = self._validate_weights(
            {**self.DEFAULT_WEIGHTS, **(weights or {})}
        )
        self.tfidf_matrix = None
        self.course_id_to_index: dict[str, int] = {}
        self.semantic_embeddings = None
        self._load_artifacts()

    @staticmethod
    def _validate_weights(weights: dict[str, float]) -> dict[str, float]:
        required = set(HybirdRecommender.DEFAULT_WEIGHTS)
        if set(weights) != required:
            missing = required - set(weights)
            extra = set(weights) - required
            raise ValueError(
                f"Weights must contain exactly {sorted(required)}. "
                f"Missing: {sorted(missing)}; unexpected: {sorted(extra)}."
            )

        try:
            validated = {name: float(value) for name, value in weights.items()}
        except (TypeError, ValueError) as error:
            raise ValueError("Recommendation weights must be numeric.") from error

        if any(not np.isfinite(value) or value < 0 for value in validated.values()):
            raise ValueError("Recommendation weights must be finite and non-negative.")
        if sum(validated.values()) <= 0:
            raise ValueError("At least one recommendation weight must be greater than zero.")
        return validated

    def _load_artifacts(self) -> None:
        logging.info("Loading hybrid recommender artifacts")
        self.tfidf_matrix = sparse.load_npz(
            self.tfidf_artifact.tfidf_matrix_file_path
        )
        self.course_id_to_index = joblib.load(
            self.tfidf_artifact.course_id_to_index_file_path
        )
        self.semantic_embeddings = np.load(
            self.semantic_artifact.embedding_file_path
        )

        if self.tfidf_matrix.shape[0] != len(self.course_id_to_index):
            raise ValueError("TF-IDF matrix and course index have different row counts.")
        if self.semantic_embeddings.shape[0] != self.tfidf_matrix.shape[0]:
            raise ValueError(
                "Semantic embeddings and TF-IDF matrix must contain the same courses."
            )

    @staticmethod
    def _minmax(values: pd.Series) -> pd.Series:
        values = pd.to_numeric(values, errors="coerce").fillna(0.0)
        minimum, maximum = values.min(), values.max()
        if minimum == maximum:
            return pd.Series(1.0, index=values.index)
        return (values - minimum) / (maximum - minimum)

    def _candidate_scores(
        self, course_id: str, candidate_k: int
    ) -> pd.DataFrame:
        if course_id not in self.course_id_to_index:
            raise ValueError(f"Course ID {course_id} not found.")
        if candidate_k < 1:
            raise ValueError("candidate_k must be at least 1.")

        row_index = self.course_id_to_index[course_id]
        semantic_scores = self.semantic_embeddings @ self.semantic_embeddings[row_index]
        tfidf_scores = cosine_similarity(
            self.tfidf_matrix[row_index], self.tfidf_matrix
        ).ravel()

        semantic_indices = np.argsort(semantic_scores)[::-1][:candidate_k + 1]
        tfidf_indices = np.argsort(tfidf_scores)[::-1][:candidate_k + 1]
        candidate_indices = set(semantic_indices) | set(tfidf_indices)
        candidate_indices.discard(row_index)

        index_to_course_id = {
            index: candidate_id
            for candidate_id, index in self.course_id_to_index.items()
        }
        rows = [
            {
                "course_id": index_to_course_id[index],
                "semantic_score": float(semantic_scores[index]),
                "tfidf_score": float(tfidf_scores[index]),
            }
            for index in candidate_indices
            if index in index_to_course_id
        ]
        return pd.DataFrame(rows)

    def recommend(
        self,
        data: pd.DataFrame,
        course_id: str,
        top_n: int = 10,
        candidate_k: int = 100,
    ) -> pd.DataFrame:
        """Return the highest-ranked hybrid recommendations for a course."""
        if top_n < 1:
            raise ValueError("top_n must be at least 1.")
        required_columns = {"course_id", "popularity_score", "rating_score"}
        missing_columns = required_columns - set(data.columns)
        if missing_columns:
            raise ValueError(
                f"Data is missing required recommendation columns: {sorted(missing_columns)}"
            )

        candidates = self._candidate_scores(course_id, candidate_k)
        if candidates.empty:
            return candidates

        candidates["semantic_score"] = self._minmax(candidates["semantic_score"])
        candidates["tfidf_score"] = self._minmax(candidates["tfidf_score"])
        score_columns = ["course_id", "popularity_score", "rating_score"]
        candidates = candidates.merge(data[score_columns], on="course_id", how="left")
        candidates["popularity_score"] = pd.to_numeric(
            candidates["popularity_score"], errors="coerce"
        ).fillna(0.0)
        candidates["rating_score"] = pd.to_numeric(
            candidates["rating_score"], errors="coerce"
        ).fillna(0.0)

        candidates["hybrid_score"] = sum(
            self.weights[name] * candidates[column]
            for name, column in (
                ("semantic", "semantic_score"),
                ("tfidf", "tfidf_score"),
                ("popularity", "popularity_score"),
                ("rating", "rating_score"),
            )
        )
        recommendations = (
            candidates.sort_values("hybrid_score", ascending=False)
            .head(top_n)
            .reset_index(drop=True)
        )

        information_columns = [
            column
            for column in ("course_id", "course_name", "level")
            if column in data.columns
        ]
        return recommendations.merge(
            data[information_columns], on="course_id", how="left"
        )

    def save(self) -> Hybird_function_Artifact:
        """Persist the configured hybrid recommender and return its artifact."""
        os.makedirs(self.config.hybird_model_dir, exist_ok=True)
        joblib.dump(self, self.config.hybird_model_file_path)
        logging.info(
            "Hybrid recommender saved to %s",
            self.config.hybird_model_file_path,
        )
        return Hybird_function_Artifact(
            hybird_model_file_path=self.config.hybird_model_file_path
        )

    hybrid_recommend = recommend
    hybrid_recommender = recommend
    hybird_recommender = recommend
        