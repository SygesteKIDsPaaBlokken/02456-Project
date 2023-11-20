#%%
# %load_ext autoreload
# %autoreload 2
# %% Imports
from collections import defaultdict
import gc
import numpy as np
import pandas as pd
import json
from tqdm import tqdm
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

from utils import compute_tf, clean_text

# %% Load data
print("[LOADING DATA] Start")
data_path = '/dtu/blackhole/1a/163226/'
corpus_path = data_path + 'collection.tsv'

documents = pd.read_csv(corpus_path, sep='\t', header=None)
documents.columns=['pid','passage']
print("[LOADING DATA] Done")
# %% TF.IDF config
stop_words = set(stopwords.words('english'))
chunk_size = 10_000 # ca. 3 GB
tfidf_cache = '/dtu/blackhole/1b/167931/tf_idf'
documents = documents.iloc[:100_000]
# %% Generate TF and IDF dicts
print("[FITTING TF.IDF] Start")
print('[Fitting] Cleaning text')
document_dict = {pid: clean_text(passage) for pid, passage in zip(documents['pid'],documents['passage'])}

print('[Fitting] Computing term frequency')
tf = {}
idf = defaultdict(lambda: 0)
for pid, doc in tqdm(document_dict.items()):
    doc_tf = defaultdict(lambda: 0)
    
    for term in doc.split():
        if term not in stop_words:
            doc_tf[term] += 1
    
    if len(doc_tf.keys()) == 0: continue
    
    max_freq = max(doc_tf.values())
    for term in doc_tf.keys():
        doc_tf[term] = doc_tf[term]/max_freq
        idf[term] += 1

    tf[pid] = dict(doc_tf)

print('[Fitting] Computing inverse document frequency')
N = len(documents)
for term in idf.keys():
    idf[term] = np.log2(N/idf[term])

vocabulary = list(idf.keys())
print("[FITTING TF.IDF] Done")
# %% Save dicts
print("[SAVING TF.IDF] Started")
with open(tfidf_cache+'/tf.json','w') as f:
    json.dump(tf, f)

with open(tfidf_cache+'/idf.json','w') as f:
    json.dump(dict(idf), f)

print("[SAVING TF.IDF] Done")
# %% Create query chunks
chunk_cache = '/dtu/blackhole/1b/167931/tf_idf/chunks'
chunk_size = 10_000
queries_eval_path = data_path + 'queries.eval.tsv'

queries_eval = pd.read_csv(queries_eval_path, sep='\t', header=None)
queries_eval.columns=['qid','query']

chunk_idx = -1
for i, (qid, query) in enumerate(tqdm(zip(queries_eval['qid'],queries_eval['query']))):
    query = clean_text(query)

    if i % chunk_size == 0:
        if i // chunk_size > 0:
            np.save(f"{chunk_cache}/qchunk_{chunk_idx}.npy", chunk)
            np.save(f"{chunk_cache}/qchunk_{chunk_idx}_pids.npy", chunk_pids)

            del chunk
            gc.collect()

        chunk = np.zeros((chunk_size, len(vocabulary)))
        chunk_pids = []
        chunk_idx += 1

    chunk_pids.append(pid)
    tf = compute_tf(query, stop_words)
    
    for term in tf.keys():
        term_idx = vocabulary.index(term)
        chunk[i, term_idx] = tf[term]*idf[term]
