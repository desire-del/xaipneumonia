from zenml.steps import step
from datasets import load_dataset
from datasets.dataset_dict import DatasetDict


@step
def ingestion(hf_link : str) -> DatasetDict:
    """Ingest data from a source."""
    
    print("Ingesting data...")
    try:
        dataset = load_dataset(hf_link)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None
    return dataset