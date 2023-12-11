import numpy as np
import pandas as pd
from time import time

from tqdm import tqdm
from numpy.linalg import norm
import nltk
from nltk.corpus import stopwords
from utils.evaluation import ScoringEvaluation


def score(A, B):
    B_ = B.T if B.shape[0] != A.shape[-1] else B
    return A@B_ / (norm(A) * norm(B_, axis=0))


def ftPhrase(passage, ftModel):
    temp = []
    for word in passage.split(' '):
        if word not in stop_words:
            temp.append(ftModel[word])
    return np.mean(temp, axis=0)


def rank(queries, queryIDs, corpus_vectors, model, name='', N=1000, ):
    try:
        out = pd.read_csv(f'data/{name}_RankingResults.csv', sep=',')
        return out
    except FileNotFoundError:
        pass
    queryColumn, passageColumn, scoreColumn = [np.empty(len(queries) * N) for _ in range(3)]

    i = 0
    for query in tqdm(queries, desc='Scoring'):
        qid = queryIDs[i]
        ftQuery = ftPhrase(query, model) if name == 'ft' else np.array(model.encode([query])).flatten()
        scores = score(ftQuery, corpus_vectors)
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
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))
    startQuery = 0
    stopQuery = 200
    corpusStop = -1  # -1 for all
    fasttext_run = True
    sbert = False
    t = time()

    # Load testing data
    dfCorpus = pd.read_csv('data/collectionReduced.tsv', sep='\t', header=None)
    dfQueries = pd.read_csv('data/msmarco-test2019-queries.tsv', sep='\t', header=None)

    corpus = dfCorpus.values[:, 1].flatten()[:corpusStop]
    corpusIDs = dfCorpus.values[:, 0].flatten()[:corpusStop].astype(np.float32)
    queries = dfQueries.values[:, 1].flatten()[startQuery:stopQuery]
    queryIDs = dfQueries.values[:, 0].flatten()[startQuery:stopQuery].astype(np.float32)

    N = 1000

    # Run evaluation
    topK = [1, 2, 3, 4, 5, 10, ]
    dfEval = pd.read_csv('data/0-3scoringTestSet.txt', sep=' ')

    if fasttext_run:
        import fasttext.util
        # fasttext.util.download_model('en', if_exists='ignore')  # English
        ft = fasttext.load_model('Fasttext/cc.en.300.bin')
        # fasttext.util.reduce_model(ft, 100)  # Reduce the dimensions to save RAM
        try:
            ftCorpus = pd.read_csv(f'data/ftCorpus.csv', index_col=0).values
        except FileNotFoundError:
            ftCorpus = np.array([ftPhrase(passage, ft) for passage in tqdm(corpus, desc='Creating corpus vectors')])
            pd.DataFrame(ftCorpus).to_csv(f'data/ftCorpus.csv')
            print(f'Calculating all FT passage vectors took {time() - t:.3f} seconds')
        dfFT = rank(queries, queryIDs, ftCorpus, ft, name='ft', N=N)
        ScoringEvaluation(dfEval, dfFT, topK, name='Fasttext')



