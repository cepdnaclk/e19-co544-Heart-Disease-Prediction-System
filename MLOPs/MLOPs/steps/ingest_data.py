import logging

import pandas as pd
import numpy as np
from zenml import step

class ImportData:
    """
    Ingest data from a given path.

    """
    
    def __init__(self, data_path: str):
        """
        Initialize the data path.
        """
        self.data_path = data_path
        
    def get_data(self):
        """
        Getting the data from the given path.
        """
        logging.info(f"Reading data from {self.data_path}")
        df = pd.read_csv(self.data_path)
        # return df
        
@step
def ingest_data(data_path: str) -> pd.DataFrame:
    """
    Ingest data from a given path.
    
    Args:
        data_path: Path to the data file.
    """
    try:
        ingest_data = ImportData(data_path)
        df = ingest_data.get_data()
        return df
    
    except Exception as e:
        logging.error(f"Failed to ingest data: {e}")
        return e
    
        
        