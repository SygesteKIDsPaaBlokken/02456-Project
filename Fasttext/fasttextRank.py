import numpy as np
import pandas as pd
import fasttext.util
from time import time
from tqdm import tqdm


def score(q, s):
    return np.sqrt(np.sum(np.square(q - s), axis=1))


if __name__ == '__main__':
    startQuery = 0
    stopQuery = 200
    corpusStop = -1     # -1 for all
    t = time()
    fasttext.util.download_model('en', if_exists='ignore')  # English
    ft = fasttext.load_model('cc.en.300.bin')
    fasttext.util.reduce_model(ft, 100)         # Reduce the dimensions to save RAM
    dfCorpus = pd.read_csv('collection.tsv', sep='\t')
    dfQueries = pd.read_csv('msmarco-test2019-queries.tsv', sep='\t')

    corpus = dfCorpus.values[:, 1].flatten()[:corpusStop]
    corpusIDs = dfCorpus.values[:, 0].flatten()[:corpusStop].astype(np.float32)
    queries = dfQueries.values[:, 1].flatten()[startQuery:stopQuery]
    queryIDs = dfQueries.values[:, 0].flatten()[startQuery:stopQuery].astype(np.float32)
    print(f'Loading data and model took {time() - t:.3f} seconds')  # ~60
    t = time()
    try:
        ftCorpus = pd.read_csv(f'ftCorpus.csv', index_col=0).values
    except FileNotFoundError:
        ftCorpus = []
        for passage in tqdm(corpus, desc='Creating corpus vectors'):
            ftCorpus.append(np.mean([ft[word] for word in passage.split(' ')], axis=0))
        ftCorpus = np.array(ftCorpus)
        pd.DataFrame(ftCorpus).to_csv(f'ftCorpusNew.csv')
    print(f'Calculating all passage vectors took {time() - t:.3f} seconds')

    N = 1000

    queryColumn, passageColumn, scoreColumn = [np.empty(len(queries) * N) for _ in range(3)]

    i = 0
    for query in tqdm(queries, desc='Scoring'):
        qid = queryIDs[i]
        ftQuery = ft[query]
        scores = score(ftQuery, ftCorpus)
        scoresArgsort = (-1 * scores).argsort()[:N]

        queryColumn[i*N:(i+1) * N] = qid
        passageColumn[i*N:(i+1) * N] = scoresArgsort    #
        scoreColumn[i*N:(i+1) * N] = scores[scoresArgsort]
        i += 1

    t = time()
    out = pd.DataFrame(np.vstack((queryColumn, np.array(passageColumn), np.array(scoreColumn))).T, columns=['query', 'passage', 'score'])
    out.to_csv(f'RankingResults_{startQuery}_{stopQuery}.csv', sep=',')
    print(f'Saving data took {time() - t:.3f} seconds')  #
