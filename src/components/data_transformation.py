import os

from src.entity.config_entity import DataTransformationConfig
from src.entity.artifact_entity import DataIngestionArtifact, DataTransformationArtifact
from src.constants import *
from src.logger import logging
from src.exception import CustomException
import re
import pandas as pd
import sys
import numpy as np


class DataTransformation:
    def __init__(self, data_ingestion_artifact: DataIngestionArtifact, data_transformation_config: DataTransformationConfig):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_transformation_config = data_transformation_config
        except Exception as e:
            raise CustomException(e, sys)

    def drop_duplicate_rows(self, data):
        try:
            logging.info("Dropping duplicate rows from the data")
            result = data.copy()
            result = result.drop_duplicates()
            logging.info("Dropped duplicate rows from the data")
            return result
        except Exception as e:
            raise CustomException(e, sys)

    def drop_columns(self, data):
        try:
            logging.info(f"Dropping the unnecessary columns: {DROP_COLUMNS}")
            result = data.copy()
            result = result.drop(columns = DROP_COLUMNS, errors="ignore")
            logging.info(f"Dropped the unnecessary columns: {DROP_COLUMNS}")
            return result
        except Exception as e:
            raise CustomException(e, sys)

    def clean_numeric_columns(self, value: object, remove_commas: bool = False) -> float:
        try:
            if pd.isna(value):
                return float("nan")
            text = str(value).replace(",", "") if remove_commas else str(value)
            match = re.search(r"\d+(?:\.\d+)?", text)
            return float(match.group()) if match else float("nan")
        except Exception as e:
            raise CustomException(e, sys)

    def clean_data(self, data:pd.DataFrame) -> pd.DataFrame:
        try:
            logging.info("Cleaning the data by handling missing values, cleaning numeric columns, and ensuring required columns are present")
            result = data.copy()
            missing = [column for column in REQUIRED_COLUMNS if column not in result.columns]
            if missing:
                    raise ValueError(f"Dataset is missing required columns: {missing}")
            
            logging.info("Cleaning numeric columns")
            for column in NUMERIC_COLUMNS:
                if column in result.columns:
                                result[column] = result[column].map(
                                     lambda value: self.clean_numeric_columns(value, remove_commas=column in COMMA_NUMERIC_COLUMNS),
                                     na_action="ignore",
                                )
            result = result[result["reviews_avg"].between(0, 5)].copy()
            for column in CATEGORICAL_COLUMNS:
                 result[column] = result[column].fillna(result[column].mode().iloc[0] if not result[column].mode().empty else "Unknown")

            for column in DROP_NA_COLUMNS:
                 result = result.dropna(subset=[column])

            for column in NUMERIC_COLUMNS:
                if column in result:
                    result[column] = result[column].fillna(result[column].median())

            result = result.reset_index(drop=True)
            result["course_id"] = [str(index).zfill(4) for index in range(1, len(result) + 1)]
            return result
        except Exception as e:
            raise CustomException(e, sys)
    
    # Feature Engineering: Min-Max Scaling for Numeric Columns
    def _min_max_scale(self, series: pd.Series) -> pd.Series:
         try:
            minimum, maximum = series.min(), series.max()
            if maximum == minimum:
                return pd.Series(0.0, index=series.index)
            return (series - minimum) / (maximum - minimum)
         except Exception as e:
            raise CustomException(e, sys)

    def feature_engineering(self,data: pd.DataFrame) -> pd.DataFrame:
        try:
            logging.info("Performing feature engineering to create new features based on existing data")
            result = data.copy()
            result["course_text"] = (
            "course: " + result["course_name"].astype(str) + " "
            + "description: " + result["course description"].astype(str) + " "
            + "instructor: " + result["instructor"].astype(str) + " "
            + "level: " + result["level"].astype(str)
            )
            log_students = np.log1p(result["students_count"].fillna(0))
            log_reviews = np.log1p(result["reviews_count"].fillna(0))
            result["students_score"] = self._min_max_scale(log_students)
            result["reviews_score"] = self._min_max_scale(log_reviews)
            result["rating_score"] = self._min_max_scale(result["reviews_avg"].fillna(result["reviews_avg"].median()))
            result["popularity_score"] = (
                0.40 * result["students_score"]
                + 0.25 * result["reviews_score"]
                + 0.25 * result["rating_score"]
            )
            return result
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self) -> DataTransformationArtifact:
         try:
            logging.info("Starting data transformation process")
            data = pd.read_csv(self.data_ingestion_artifact.data_file_path)

            data = self.drop_duplicate_rows(data)
            data = self.drop_columns(data)
            logging.info("Droped the duplicate rows and unnecessary columns from the data")

            data = self.clean_data(data)
            logging.info("Cleaned the data by handling missing values, cleaning numeric columns, and ensuring required columns are present")

            data = self.feature_engineering(data)
            logging.info("Performed feature engineering to create new features based on existing data")

            os.makedirs(self.data_transformation_config.transformed_data_dir, exist_ok=True)
            transformed_data_file_path = os.path.join(self.data_transformation_config.transformed_data_dir,"transformed_data.csv")
            data.to_csv(transformed_data_file_path, index=False)
            logging.info(f"Saved transformed data to {transformed_data_file_path}")

            return DataTransformationArtifact(transformed_data_file_path=transformed_data_file_path)
         except Exception as e:
            raise CustomException(e, sys)