#%%
%load_ext autoreload
%autoreload 2
# %% Imports
import pandas as pd
import nltk

nltk.download('stopwords')

from Encoder import TFIDF
from Loader import clean_text

# %% Load data
print("[LOADING DATA] Start")
data_path = '/dtu/blackhole/1a/163226/'
corpus_path = data_path + 'collection.tsv'
queries_eval_path = data_path + 'queries.eval.tsv'

documents = pd.read_csv(corpus_path, sep='\t', header=None)
documents.columns=['pid','passage']

queries_eval = pd.read_csv(queries_eval_path, sep='\t', header=None)
queries_eval.columns=['qid','query']
print("[LOADING DATA] Done")
# %% Fit TF.IDF
print("[FITTING TF.IDF] Start")
tfidf = TFIDF(documents.iloc[:500_000])
print("[FITTING TF.IDF] Done")
# %% Test
print("[TEST QUERY] Start")
query = "Manhattan"

results = tfidf.search(query)
print(results)
print("[TEST QUERY] DONE")
# %% Evaluation
evaluation = pd.DataFrame({}, columns=['qid','pid','score'])
for qid, query in zip(queries_eval['qid'],queries_eval['query']):
    results = tfidf.search(query, 100)
    results = pd.DataFrame(results, columns=['pid','score'])
    results = results.sort_values('score')
    results['qid'] = qid

    pd.concat([evaluation,results])


