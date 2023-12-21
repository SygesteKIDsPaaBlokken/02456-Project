import numpy as np
import pandas as pd
from tqdm import tqdm
from time import time
from rank_bm25 import BM25Okapi


def rank_bm25(bm25, queries, queryIDs, N=1000, name='bm25'):
    try:
        out = pd.read_csv(f'data/{name}_RankingResults.csv', sep=',')
        return out
    except FileNotFoundError:
        pass
    queryColumn, passageColumn, scoreColumn = [np.empty(len(queries) * N) for _ in range(3)]

    i = 0
    for query in tqdm(queries, desc=name):  # ~3 minutes per query

        query_tokenized = query.split(' ')
        scores = np.array(bm25.get_scores(query_tokenized))

        qid = queryIDs[i]
        scoresArgsort = (-1 * scores).argsort()[:N]

        queryColumn[i * N:(i + 1) * N] = qid
        passageColumn[i * N:(i + 1) * N] = scoresArgsort  #
        scoreColumn[i * N:(i + 1) * N] = scores[scoresArgsort]
        i += 1

    t = time()
    out = pd.DataFrame(np.vstack((queryColumn, np.array(passageColumn), np.array(scoreColumn))).T,
                       columns=['query', 'passage', 'score'])
    out.to_csv(f'data/{name}_RankingResults.csv', sep=',')
    print(f'Saving {name} data took {time() - t:.3f} seconds')
    return out


if __name__ == '__main__':
    corpusStop = -1
    # Load testing data
    dfCorpus = pd.read_csv('data/collectionReduced.tsv', sep='\t', header=None)
    dfQueries = pd.read_csv('data/msmarco-test2019-queries.tsv', sep='\t', header=None)
    dfEval = pd.read_csv('data/0-3scoringTestSet.txt', sep=' ')

    qids = np.array(
        [dfQueries.values[dfQueries.values[:, 0] == qid, 0] for qid in np.unique(dfEval['query'])]).squeeze()
    queries = np.array(
        [dfQueries.values[dfQueries.values[:, 0] == qid, 1] for qid in np.unique(dfEval['query'])]).squeeze()

    corpus_tokenized = [str(passage).split(' ') for passage in tqdm(dfCorpus.values[:, 1])]
    t = time()
    model = BM25Okapi(corpus_tokenized, k1=0.60, b=0.62)
    print(f'Loading corpus into BM25 took {t-time():.2f} seconds')
    rank_bm25(model, queries, qids)
