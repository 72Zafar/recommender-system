import os
from dataclasses import dataclass
from src.constants import *
from datetime import datetime

@dataclass
class DataIngestionArtifact:
    data_file_path: str

    
@dataclass
class DataTransformationArtifact:
    transformed_data_file_path: str
    
@dataclass 
class TFIDFVectorizerArtifact:
    tfidf_vectorizer_file_path: str
    tfidf_matrix_file_path: str
    course_id_to_index_file_path: str