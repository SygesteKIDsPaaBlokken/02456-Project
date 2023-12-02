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
    

class MSMarcoSmallList(Dataset):
    def __init__(self, qidpidtriples: Path, limit: int = None) -> None:
        self.qidpidtriples = []
        self._load_data_from_tsv_file(qidpidtriples, limit)

    def _load_data_from_tsv_file(self, qidpidtriples: Path, limit: int = None):
        with open(qidpidtriples, newline='', encoding='utf-8') as csvfile:
            qidpidtriples_reader = csv.reader(csvfile, delimiter='\t')
            self._save_qidpidtriples_from_csv_reader(qidpidtriples_reader, limit)
        
    def _save_qidpidtriples_from_csv_reader(self, csv_reader, limit: int = None) -> None:
        for i, qidpidtriple in enumerate(csv_reader):
            if limit is not None and i > limit:
                break
            self._save_qidpidtriple(qidpidtriple)

    def _save_qidpidtriple(self, qidpidtriple: tuple[str]) -> None:
        self.qidpidtriples.append(
            InputExample(texts=qidpidtriple)
        )
    
    def __len__(self):
        return len(self.qidpidtriples)

    def __getitem__(self, idx):
        if (idx == len(self)):
            raise StopIteration
        
        qidpidtriple = self.qidpidtriples[idx]

        return InputExample(texts=qidpidtriple)