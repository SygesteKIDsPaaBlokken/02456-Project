from time import perf_counter
from pathlib import Path
import os
import sys

import torch
from torch.utils.data import DataLoader
from sentence_transformers import losses

from MSMarcoDatasetSmall import MSMarcoSmallPandas, MSMarcoSmallList
from config import DEVICE, TRIPLES_SMALL_PATH, WARMUP_STEPS, USE_AMP, BATCH_SIZE, NUM_WORKERS

sys.path.append('../02456-Project')
from models.SBERT import SBERT

NUM_WORKERS = len(os.sched_getaffinity(0)) # number of avaliable CPU cores

DATASET_LIMIT = 1_000_000

load_times = {'pandas': 0.0, 'list': 0.0}
datasets = [(MSMarcoSmallPandas, 'pandas'), (MSMarcoSmallList, 'list')]

for dataset_class, dataset_type in datasets + list(reversed(datasets)):
    start_time = perf_counter()
    
    dataset = dataset_class(TRIPLES_SMALL_PATH)

    end_time = perf_counter()
    total_time = end_time - start_time
    load_times[dataset_type] += total_time

print(load_times)
print('-'*100)

train_times = {'pandas': 0.0, 'list': 0.0}
for dataset_class, dataset_type in datasets + list(reversed(datasets)):
    dataset = dataset_class(TRIPLES_SMALL_PATH, limit=DATASET_LIMIT)
    dataloader = DataLoader(
        dataset,
        shuffle=True, 
        batch_size=BATCH_SIZE, 
        num_workers=NUM_WORKERS
    )

    SBERT_model = SBERT()
    model = SBERT_model.model
    model.to(DEVICE)

    loss = losses.MultipleNegativesRankingLoss(model=model)

    start_time = perf_counter()
    model.fit(
        train_objectives=[(dataloader, loss)],
        epochs=1,
        warmup_steps=WARMUP_STEPS,
        use_amp=USE_AMP,
        show_progress_bar=False,
        save_best_model=False,
    )
    end_time = perf_counter()

    total_time = end_time - start_time
    load_times[dataset_type] += total_time

print(load_times)
