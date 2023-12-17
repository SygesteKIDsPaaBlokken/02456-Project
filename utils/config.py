import os
from pathlib import Path

import torch

from utils.batch_size import get_batch_size_from_vram

# Settings
## Machine used
LOCAL = False

## PyTorch
USE_CUDA = True
DEVICE = 'cuda' if torch.cuda.is_available() and USE_CUDA else 'cpu'
EPOCHS = 1
BATCH_SIZE = get_batch_size_from_vram() if DEVICE == 'cuda' else 128
NUM_WORKERS = len(os.sched_getaffinity(0)) # number of avaliable CPU cores
WARMUP_STEPS = 1_000

## Data
DATA_FOLDER = Path('/dtu/blackhole/1a/163226') if not LOCAL else Path(os.getcwd())
TRIPLES_SMALL_PATH = DATA_FOLDER / 'triples.train.small.tsv'

MSMARCO_PATH = Path('data/MSMarco')
RANKING_PATH = Path('data/rankings')
EVALUATION_PATH = Path('data/evaluations')

## Models
SBERT_MODELS_PATH = Path('/dtu/blackhole/1b/167931/SBERT_models')

## Model settings
MODEL_NAME = 'HP_SBERT'
USE_AMP = True
VERBOSE = True
SAVE_MODEL = True