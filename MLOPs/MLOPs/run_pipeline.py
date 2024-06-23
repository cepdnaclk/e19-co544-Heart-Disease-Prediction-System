from zenml import pipelines

@pipelines.Pipeline()
def training_pipeline(data_path: str) -> None:
    """
    Training pipeline.
    
    Args:
        data_path: Path to the data file.
    """
    pass
    # ingest_data(data_path)
    # clean_data()
    # train_model()
    # evaluate_model()