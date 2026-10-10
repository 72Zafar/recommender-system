import os
from dataclasses import dataclass
from src.constants import *
from datetime import datetime

@dataclass
class DataIngestionArtifact:
    data_file_path: str
    