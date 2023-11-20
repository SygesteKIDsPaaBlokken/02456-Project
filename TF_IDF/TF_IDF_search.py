# %% Imports
from collections import defaultdict
import json
from tqdm import tqdm
import pandas as pd
import numpy as np
import gc
from sklearn.metrics.pairwise import cosine_similarity
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

from Loader import clean_text
# %% Config
data_path = '/dtu/blackhole/1a/163226/'
queries_eval_path = data_path + 'queries.eval.tsv'
stop_words = set(stopwords.words('english'))
tfidf_cache = '/dtu/blackhole/1b/167931/tf_idf'
# %% Load TF and IDF dicts
with open(tfidf_cache+'/idf.json', 'r') as f:
    idf = json.load(f)
with open(tfidf_cache+'/tf.json', 'r') as f:
    tf = json.load(f)

# %% Load queries
queries_eval = pd.read_csv(queries_eval_path, sep='\t', header=None)
queries_eval.columns=['qid','query']

# %% Prepare search
k = 10
results = {qid: {i: 0 for i in range(k)} for qid in queries_eval['qid']}
result_keys = {qid: {'minKey':0, 'minValue':0} for qid in queries_eval['qid']}

vocabulary = list(idf.keys())
chunk_size = 10_000 # ca. 3 GB

def compute_tf(query: str):
    tf = defaultdict(lambda: 0)
    for term in query.split():
        if term not in stop_words:
            tf[term] += 1
    
    if len(tf) == 0:
        return tf
    
    max_freq = max(tf.values())
    for term in tf.keys():
        tf[term] = tf[term]/max_freq

    return tf
    
def compute_feature_vector(tf:dict):
    global vocabulary, idf
    doc_feature_vector = np.zeros(len(vocabulary))  

    for term in tf.keys():
        if term not in vocabulary: continue

        idx = vocabulary.index(term)
        doc_feature_vector[idx] = tf[term]*idf[term]

    return doc_feature_vector

def search_in_chunk(chunk: np.ndarray, pids: list[int]):
    global results, queries_eval
    
    for qid, query in zip(queries_eval['qid'],queries_eval['query']):
        query = clean_text(query)
        tf = compute_tf(query)
        vec = compute_feature_vector(tf).reshape(1,-1)
        q_keys = result_keys[qid]
        q_results = results[qid]

        similarities = cosine_similarity(chunk, vec)
        for i, similarity in enumerate(similarities):
            if similarity > q_keys['minValue']:
                del q_results[q_keys['minKey']]
                pid = pids[i]
                q_results[pid] = similarity
                q_keys['minKey'] = min(q_results, key=q_results.get)
                q_keys['minValue'] = q_results[q_keys['minKey']]

        del similarities
        gc.collect()


# %% Search
for i, pid in enumerate(tqdm(tf.keys())):
    if i % chunk_size == 0:
        if i // chunk_size > 0:
            search_in_chunk(chunk, chunk_pids)

            del chunk
            gc.collect()

        chunk = np.zeros((chunk_size, len(vocabulary)))
        chunk_pids = []

    chunk_pids.append(pid)
    doc_tf = tf[pid]
    
    for term in doc_tf.keys():
        term_idx = vocabulary.index(term)
        chunk[i, term_idx] = doc_tf[term]*idf[term]


# %% Evaluation
evaluation = pd.DataFrame({}, columns=['qid','pid','score'])
for qid in queries_eval['qid']:
    q_results = pd.DataFrame(results[qid], columns=['pid','score'])
    q_results = q_results.sort_values('score')
    q_results['qid'] = qid

    pd.concat([evaluation,results])
# %% Save evaluation
evaluation.to_csv(tfidf_cache+'/evaluations.csv', index=False)