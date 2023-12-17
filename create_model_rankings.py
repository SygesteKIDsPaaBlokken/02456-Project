# %% Jupyter extensions
# %load_ext autoreload
# %autoreload 2
# %% Imports
from time import time
import pandas as pd
from argparse import ArgumentParser

from utils import MODEL, get_SBERT_model_path
from utils.evaluation import evaluate_model
from utils.ranking import rank_sbert, rank_fuzzy
from utils.config import VERBOSE, MSMARCO_PATH

# %% Select model
model:MODEL = MODEL.FUZZY
topK:int = 100
# %% CMD arguments
parser = ArgumentParser()
parser.add_argument('--model', default=model, type=MODEL, choices=list(MODEL), )
parser.add_argument('-k', default=topK, type=int, help='Amount of passages to find per query')

options = parser.parse_args()

# Settings
model:MODEL = options.model
topK:int = options.k
# %% Loading data
if VERBOSE:
    data_loading_start = time()
    print("########### Loading data ###########")
passages = pd.read_csv(MSMARCO_PATH.joinpath('collectionReduced.tsv'), sep='\t', header=None)
passages.columns = ['pid','passage']
if VERBOSE: 
    queries_loading_start = time()
    print(f"Passages loaded ({(queries_loading_start - data_loading_start)/60:.2f} min)")

queries = pd.read_csv(MSMARCO_PATH.joinpath('msmarco-test2019-queries.tsv'), sep='\t', header=None)
queries.columns = ['qid','query']
if VERBOSE: 
    print(f"Queries loaded ({(time() - queries_loading_start)/60:.2f} min)")
    print(f"########### Data loaded ({(time() - data_loading_start)/60:.2f} min) ###########")

# %% Ranking passages
if VERBOSE:
    ranking_start = time()
    print("########### Ranking passages ###########")

if 'SBERT' in model.value:
    rankings = rank_sbert(
        queries,
        passages,
        model_path=get_SBERT_model_path(model),
        model_name=model.value
    )
elif model == MODEL.FUZZY:
    rankings = rank_fuzzy(
        queries,
        passages,
        max_l_dist=1
    )

if VERBOSE:
    print(f"########### Ranking done ({(time()-ranking_start)/60:.2f} min) ###########")