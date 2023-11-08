import numpy as np
import pandas as pd
import fasttext.util
from time import time
from tqdm import tqdm


def score(q, s):
    return np.sqrt(np.sum(np.square(q - s)))


if __name__ == '__main__':
    startQuery = 0
    stopQuery = 200000
    corpusStop = -1     # -1 for all
    t = time()
    fasttext.util.download_model('en', if_exists='ignore')  # English
    ft = fasttext.load_model('cc.en.300.bin')
    fasttext.util.reduce_model(ft, 100)         # Reduce the dimensions to save RAM
    dfCorpus = pd.read_csv('collection.tsv', sep='\t')
    dfQueries = pd.read_csv('queries.train.tsv', sep='\t')

    corpus = dfCorpus.values[:, 1].flatten()[:corpusStop]
    corpusIDs = dfCorpus.values[:, 0].flatten()[:corpusStop].astype(np.float32)
    queries = dfQueries.values[:, 1].flatten()[startQuery:stopQuery]
    queryIDs = dfQueries.values[:, 0].flatten()[startQuery:stopQuery].astype(np.float32)
    print(f'Loading data and model took {time() - t:.3f} seconds')  # ~60
    t = time()
    ftCorpus = np.array([np.mean([ft[word] for word in passage[0].split(' ')]) for passage in corpus])
    print(f'Calculating all passage vectors took {time() - t:.3f} seconds')
    N = 1000

    queryColumn = []
    passageColumn = []
    scoreColumn = []

    i = 0
    for query, qid in tqdm(zip(queries, queryIDs), desc='Scoring'):
        ftQuery = ft[query]
        scores = np.array([score(ftQuery, ftPassage) for ftPassage in ftCorpus])
        scoresArgsort = (-1 * scores).argsort()[:N]

        queryColumn += [qid for _ in range(N)]
        passageColumn += list(scoresArgsort)    #
        scoreColumn += list(scores[scoresArgsort])
        i += 1

    t = time()
    out = pd.DataFrame(np.vstack((np.array(queryColumn), np.array(passageColumn), np.array(scoreColumn))).T, columns=['query', 'passage', 'score'])
    out.to_csv(f'RankingResults_{startQuery}_{stopQuery}.csv', sep='\t')
    print(f'Saving data took {time() - t:.3f} seconds')  #
