from zenml import step
import datasets as ds
from datasets import load_dataset
import re
import logging
from typing import Tuple , Union , Any
from datasets import Dataset, DatasetDict, IterableDataset, IterableDatasetDict

@step
def ingest_data()->Tuple[Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset], int]:
    logging.info("Loading the dataset from Datasets .....")
    ds = load_dataset("Trelis/tiny-shakespeare")
    logging.info("Dataset loaded !")
    maxInDs = 0
    logging.info("Calculating max length ......")
    for x in ds['train']:
        x:Any = x 
        sentence_array = re.findall(r"\b\w+(?:'\w+)?\b|[.!?,-]", x['Text'].lower())
        maxInDs = max(len(sentence_array),maxInDs)
    return ds , maxInDs

