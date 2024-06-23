import logging
import pandas as pd

from zenml import step

@step
def clean_data(df: pd.DataFrame) -> None:
    """
    Clean the data.
    
    Args:
        df: Dataframe to be cleaned.
    """
    pass
    # try:
    #     logging.info("Cleaning data.")
    #     # Dropping all rows with NaN values
    #     df = df.dropna()
        
    #     # Dropping all duplicate rows
    #     df = df.drop_duplicates()
        
    #     return df
        
    # except Exception as e:
    #     logging.error(f"Failed to clean data: {e}")
    #     return e