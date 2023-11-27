#%% imports
from sentence_transformers import SentenceTransformer, models, losses
from torch.utils.data import DataLoader
import torch
from pathlib import Path

from utils.MSMarcoDataset import MSMarco
from models.SBERT import SBERT
# %% Setup cuda
SBERT_model = SBERT()
model = SBERT_model.model

use_cuda = True
device = 'cuda' if torch.cuda.is_available() and use_cuda else 'cpu'
model.to(device)
print(device)

#%% Setup loss
train_loss = losses.MultipleNegativesRankingLoss(model)

#%% Get data
data_path = Path('/dtu/blackhole/1a/163226')

qidpidtriples_path = data_path / 'qidpidtriples.train.full.2.tsv'
queries_path = data_path / 'queries.train.tsv'
passages_path = data_path / 'collection.tsv'
ms_marco_dataset = MSMarco(qidpidtriples_path, queries_path, passages_path)
train_dataloader = DataLoader(ms_marco_dataset, shuffle=True, batch_size=128, num_workers=4)

#%% train model
# out_path = Path('/zhome/12/a/163226/Desktop/02456/02456-Project/trained_models')

model.fit(
    train_objectives=[(train_dataloader, train_loss)], 
    epochs=5, 
    warmup_steps=100, 
    show_progress_bar=True,
    output_path='trained_models/1',
    save_best_model=True,
    )
