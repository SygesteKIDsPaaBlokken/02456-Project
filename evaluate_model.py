# %% Jupyter extensions
# %load_ext autoreload
# %autoreload 2
# %% Imports
from time import time
import pandas as pd
from utils.evaluation import evaluate_model
from utils.config import EVALUATION_PATH, VERBOSE, MSMARCO_PATH, RANKING_PATH
from utils import MODEL

# %% Set model
model:MODEL = MODEL.SBERT_1e
topK:int = 100
# %% Loading rankings
ranking_file = f"{model.value}_top{topK}.csv"
model_passage_rankings = pd.read_csv(RANKING_PATH.joinpath(ranking_file))
# %% Evaluate model
if VERBOSE:
    ranking_start = time()
    print("########### Evaluating model ###########")
top_k = [1, 2, 3, 4, 5, 10, ]
evaluation_ds = pd.read_csv(MSMARCO_PATH.joinpath('2019qrels-pass.txt'), sep=' ')

evaluate_model(
    evaluation_ds,
    model_passage_rankings,
    top_k=top_k,
    save_path=EVALUATION_PATH.joinpath(f"{model.value}.csv")
)
if VERBOSE:
    print(f"########### Evaluation completed ({(time()-ranking_start)/60:.2f} min) ###########")
# %%
