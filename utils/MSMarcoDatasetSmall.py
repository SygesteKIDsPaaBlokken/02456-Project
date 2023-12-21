from pathlib import Path
import csv

import pandas as pd
from torch.utils.data import Dataset
from sentence_transformers import InputExample

class MSMarcoSmallPandas(Dataset):
    def __init__(self, qidpidtriples: Path, limit: int = None):
        self.qidpidtriples = pd.read_csv(qidpidtriples, sep='\t', header=None, encoding='utf-8', nrows=limit)
        self.qidpidtriples.columns = ['query', 'positive', 'negative']

    def __len__(self):
        return len(self.qidpidtriples)

    def __getitem__(self, idx):
        if (idx == len(self)):
            raise StopIteration
        
        qidpidtriple = self.qidpidtriples.loc[idx]

        query = qidpidtriple['query']
        postive_passage = qidpidtriple['positive']
        negative_passage = qidpidtriple['negative']

        return InputExample(texts=[query, postive_passage, negative_passage])