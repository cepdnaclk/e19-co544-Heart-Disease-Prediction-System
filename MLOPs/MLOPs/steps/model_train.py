import logging

import pandas as pd

from zenml import step

@step

def train_model(df: pd.DataFrame) -> None:
    """
    Train the model.
    
    Args:
        df: Dataframe to train the model on.
    """
    pass
    # try:
    #     logging.info("Training the model.")
    #     # Splitting the data into features and target
    #     X = df.drop(columns=['target'])
    #     y = df['target']
    #     
    #     # Training the model
    #     model = Model()
    #     model.train(X, y)
    #     
    #     return model
    #     
    # except Exception as e:
    #     logging.error(f"Failed to train model: {e}")
    #     return e