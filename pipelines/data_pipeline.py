from zenml.pipelines import pipeline


def data_pipeline(
    ingestion_step,
    preprocessing_step
):
    """Define the data pipeline."""
    
    # Ingest data
    dataset = ingestion_step()
    
    # Preprocess data
    preprocessed_data = preprocessing_step(dataset=dataset)
    
    