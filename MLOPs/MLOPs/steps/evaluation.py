import logging

import pandas as pd

from zenml import step

def evaluate_model(df:pd.DataFrame) -> None:
    """
    Evaluate the model.
    
    Args:
        df: Dataframe to evaluate the model on.
    """
    pass
    # try:
    #     logging.info("Evaluating the model.")
    #     # Splitting the data into features and target
    #     X = df.drop(columns=['target'])
    #     y = df['target']
    #     
    #     # Evaluating the model
    #     model = Model()
    #     model.evaluate(X, y)
    #     
    #     return model
    #     
    # except Exception as e:
    #     logging.error(f"Failed to evaluate model: {e}")
    #     return e