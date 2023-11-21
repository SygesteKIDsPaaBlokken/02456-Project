import pandas as pd
from torch.utils.data import Dataset
from pathlib import Path
from sentence_transformers import InputExample

class MSMarco(Dataset):
    def __init__(self, qidpidtriples: Path, queries: Path, passages: Path):
        self.qidpidtriples = pd.read_csv(qidpidtriples, sep='\t', header=None)
        self.qidpidtriples.columns = ['qid', 'ppid', 'npid']

        self.queries = pd.read_csv(queries, sep='\t', header=None, index_col=0)
        self.queries.columns = ['query']

        self.passages = pd.read_csv(passages, sep='\t', header=None, index_col=0)
        self.passages.columns = ['passage']

    def __len__(self):
        return len(self.qidpidtriples)

    def __getitem__(self, idx):
        if (idx == len(self)):
            raise StopIteration
        
        qidpidtriple = self.qidpidtriples.loc[idx]

        query = self.queries.loc[qidpidtriple['qid']]['query']
        postive_passage = self.passages.loc[qidpidtriple['ppid']]['passage']
        negative_passage = self.passages.loc[qidpidtriple['npid']]['passage']

        return InputExample(texts=[query, postive_passage, negative_passage])