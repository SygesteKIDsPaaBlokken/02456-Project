import numpy as np
import pandas as pd
from tqdm import tqdm
from time import time
import nltk
from nltk.corpus import stopwords
from utils.evaluation import ScoringEvaluation
from sentence_transformers import SentenceTransformer, util
from utils.config import DEVICE, DATA_FOLDER


def rank_sbert(queries, queryIDs, passages, N=1000, name='sbert'):
    MODEL_PATH = '1epoch'
    print("Using device:", DEVICE)

    model = SentenceTransformer(DATA_FOLDER + MODEL_PATH, device=DEVICE)
    
    print('#'*5, 'Encoding', '#'*5)
    SBERTCorpus = model.encode(passages, batch_size=32, show_progress_bar=True, device=DEVICE) 
    print('#'*5, 'Encoding finished', '#'*5)

    query_embeddings = model.encode(queries, batch_size=32, show_progress_bar=True, device=DEVICE) 
    
    ranking = util.semantic_search(query_embeddings, SBERTCorpus, top_k=1000)
    
    # Transform
    df_list = []
    query_id = 0
    for query_results in tqdm(ranking, desc='Transforming'):
        for result in query_results:
            corpus_id = result['corpus_id']
            score = result['score']
            df_list.append({'query': queryIDs[query_id], 'passage': df_passages.values[corpus_id,0], 'score': score})
        query_id += 1
        
    df = pd.DataFrame(df_list)
    df.to_csv(f'data/{name}_RankingResults.csv')
    


    return df
    

    t = time()
    
    print(f'Saving {name} data took {time() - t:.3f} seconds')
    return out

def remove_stop_words(words, stop_words):
    passages_clean = []
    for passage in tqdm(words):  # ~20
        temp = ''
        for word in passage.split(' '):
            if word not in stop_words:
                temp += word + ' '
        passages_clean.append(temp[:-1])
    return passages_clean

if __name__ == '__main__':
    t = time()
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))
    df_passages = pd.read_csv('data/collectionReduced.tsv', sep='\t', header=None)
    dfQueries = pd.read_csv('data/msmarco-test2019-queries.tsv', sep='\t', header=None)
    dfEval = pd.read_csv('data/0-3scoringTestSet.txt', sep=' ')
    qids = np.array(
        [dfQueries.values[dfQueries.values[:, 0] == qid, 0] for qid in np.unique(dfEval['query'])]).squeeze()
    queries = np.array(
        [dfQueries.values[dfQueries.values[:, 0] == qid, 1] for qid in np.unique(dfEval['query'])]).squeeze()

    print(f'Loading data took {time() - t:.3f} seconds')  # ~6
    passages_clean = remove_stop_words(df_passages.values[:, 1], stop_words)  # ~20
    queries_clean = remove_stop_words(queries, stop_words)
    t = time()

    df_sbert = rank_sbert(queries_clean, qids, df_passages.values[:, 1], )
    #df_sbert = pd.read_csv('SEMANTIC_SEARCH.csv')
    print(f'SBERT searching took {(time() - t) / 60:.3f} minutes')  # ~

    topK = [1, 2, 3, 4, 5, 10, ]
    dfEval = pd.read_csv('data/0-3scoringTestSet.txt', sep=' ')
    ScoringEvaluation(dfEval, df_sbert, topK=topK, name='sbert')
    pass
