#%% imports
from sentence_transformers import SentenceTransformer, models, losses
from torch.utils.data import DataLoader
import torch
from pathlib import Path

from utils.MSMarcoDataset import MSMarco
from utils.config import DATA_FOLDER, DEVICE, BATCH_SIZE, NUM_WORKERS, EPOCHS, USE_AMP, SAVE_MODEL, WARMUP_STEPS, VERBOSE
from models.SBERT import SBERT
import os

reduced = True

# %% Setup cuda
SBERT_model = SBERT()
model = SBERT_model.model

model.to(DEVICE)

#%% Setup loss
train_loss = losses.MultipleNegativesRankingLoss(model)

#%% Get data

qidpidtriples_path = DATA_FOLDER / f'triplets{"Reduced" if reduced else ""}.tsv'
queries_path = DATA_FOLDER / f'queries.train{"Reduced" if reduced else ""}.tsv'
passages_path = DATA_FOLDER / f'collection{"Reduced" if reduced else ""}.tsv'
ms_marco_dataset = MSMarco(qidpidtriples_path, queries_path, passages_path)
train_dataloader = DataLoader(ms_marco_dataset, shuffle=True, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

# %% Save settings
blackhole = '/dtu/blackhole/1b/167931/'
output_path = blackhole + "SBERT_models/3epochs"
checkpoint_path = blackhole + "SBERT_models/3epochs_checkpoints"

#%% train model

model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs = EPOCHS,
    warmup_steps = WARMUP_STEPS,
    use_amp = USE_AMP,
    show_progress_bar = VERBOSE,
    output_path = output_path,
    save_best_model = SAVE_MODEL,
    checkpoint_path= checkpoint_path
)