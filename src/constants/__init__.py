import os 
from datetime import datetime

PIPELINE_NAME:str = ""
ARTIFACTS_DIR: str = "artifacts"



FILE_NAME: str = "data.csv"


""" 
Data ingestion related constant start with DATA_INGESTION VAR NAME
"""
DATA_INGESTION_DIR_NAME: str = "data_ingestion"
DATA_INGESTION_INGESTED_DIR: str = "ingested"
DATA_INGESTION_COLLECTION_NAME: str = "course_data"

"""
    COLUMNS NAME related constant start with COLUMNS VAR NAME
"""
NUMERIC_COLUMNS = [
    "reviews_avg",
    "reviews_count",
    "course_duration",
    "lectures_count",
    "price_after_discount",
    "main_price",
    "students_count",
]

CATEGORICAL_COLUMNS = ["course_name",
                       "instructor",
                       "level"
]

DROP_COLUMNS = ["Unnamed: 14",
                "Unnamed: 15",
                "Unnamed: 16",
                "Unnamed: 17",
                "course_flag"
]

# DROP_NA columns name
DROP_NA_COLUMNS =["course url"
                , "course image",
                "course description"
]


COMMA_NUMERIC_COLUMNS = {"reviews_count", "price_after_discount", "main_price", "students_count"}

REQUIRED_COLUMNS = [
    "course_name",
    "instructor",
    "course url",
    "course image",
    "course description",
    "level",
]

"""
Data Transformation related constant start with DATA_TRANSFORMATION VAR NAME
"""

DATA_TRANSFORMATION_DIR_NAME: str = "data_transformation"
DATA_TRANSFORMATION_TRANSFORMED_DIR: str = "transformed"