# %% Jupyter extensions
# %load_ext autoreload
# %autoreload 2
# %% Imports
from time import time
import pandas as pd
from utils.evaluation import evaluate_model
from utils.ranking import rank_sbert
from utils.config import VERBOSE

# %% Loading data
if VERBOSE:
    data_loading_start = time()
    print("########### Loading data ###########")
passages = pd.read_csv('data/collectionReduced.tsv', sep='\t', header=None)
passages.columns = ['pid','passage']
if VERBOSE: 
    queries_loading_start = time()
    print(f"Passages loaded ({(queries_loading_start - data_loading_start)/60:.2f} min)")

queries = pd.read_csv('data/msmarco-test2019-queries.tsv', sep='\t', header=None)
queries.columns = ['qid','query']
if VERBOSE: 
    print(f"Queries loaded ({(time() - queries_loading_start)/60:.2f} min)")
    print(f"########### Data loaded ({(time() - data_loading_start)/60:.2f} min) ###########")

# %% Ranking passages
if VERBOSE:
    ranking_start = time()
    print("########### Ranking passages ###########")
HP_SBERT_rankings = rank_sbert(
    queries,
    passages,
    model_path='/dtu/blackhole/1b/167931/SBERT_models/HP_model'
)
if VERBOSE:
    print(f"########### Ranking done ({(time()-ranking_start)/60:.2f} min) ###########")

# %% Evaluate model
if VERBOSE:
    ranking_start = time()
    print("########### Evaluating model ###########")
top_k = [1, 2, 3, 4, 5, 10, ]
eval_ds = pd.read_csv('data/2019qrels-docs.txt', sep=' ')
evaluate_model(
    eval_ds,
    HP_SBERT_rankings,
    top_k=top_k
)
if VERBOSE:
    print(f"########### Evaluation completed ({(time()-ranking_start)/60:.2f} min) ###########")