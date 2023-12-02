import numpy as np
import pandas as pd
import fasttext.util
from time import time

from tqdm import tqdm
from numpy.linalg import norm
import nltk
from nltk.corpus import stopwords
from utils.evaluation import ScoringEvaluation, top1000Evaluation


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
    dfCorpus = pd.read_csv('data/collection.tsv', sep='\t')
    dfQueries = pd.read_csv('data/msmarco-test2019-queries.tsv', sep='\t')

    corpus = dfCorpus.values[:, 1].flatten()[:corpusStop]
    corpusIDs = dfCorpus.values[:, 0].flatten()[:corpusStop].astype(np.float32)
    queries = dfQueries.values[:, 1].flatten()[startQuery:stopQuery]
    queryIDs = dfQueries.values[:, 0].flatten()[startQuery:stopQuery].astype(np.float32)
    if fasttext_run:
        fasttext.util.download_model('en', if_exists='ignore')  # English
        ft = fasttext.load_model('Fasttext/cc.en.300.bin')
        fasttext.util.reduce_model(ft, 100)  # Reduce the dimensions to save RAM
    # Load SBERT:
    if sbert:
        from sentence_transformers import SentenceTransformer
        SBERT = SentenceTransformer('trained_models/1')

    print(f'Loading data and model took {time() - t:.3f} seconds')  # ~60
    t = time()
    # Fasttext vectors
    if fasttext_run:
        try:
            ftCorpus = pd.read_csv(f'data/ftCorpus.csv', index_col=0).values
        except FileNotFoundError:
            ftCorpus = np.array([ftPhrase(passage, ft) for passage in tqdm(corpus, desc='Creating corpus vectors')])
            pd.DataFrame(ftCorpus).to_csv(f'data/ftCorpusStop.csv')
        print(f'Calculating all FT passage vectors took {time() - t:.3f} seconds')
        t = time()
    # SBERT vectors
    if sbert:
        try:
            SBERTCorpus = pd.read_csv(f'data/SBERTCorpus.csv', index_col=0).values
        except FileNotFoundError:
            SBERTCorpus = SBERT.encode(corpus)
            pd.DataFrame(ftCorpus).to_csv(f'data/SBERTCorpus.csv')
    print(f'Calculating all SBERT passage vectors took {time() - t:.3f} seconds')

    N = 1000
    if fasttext_run:
        dfFT = rank(queries, queryIDs, ftCorpus, ft, name='ft', N=N)
    if sbert:
        dfSBERT = rank(queries, queryIDs, SBERTCorpus, SBERT, name='SBERT', N=N)
    # Run evaluation
    topK = [1, 5, 10, 100]
    dfEval = pd.read_csv('data/0-3scoringTestSet.txt', sep=' ')
    dfTop = pd.read_csv('data/msmarco-passagetest2019-top1000.tsv', sep='\t', header=None)

    if fasttext_run:
        ScoringEvaluation(dfEval, dfFT, topK, name='Fasttext')
        top1000Evaluation(dfTop, dfFT, topK, name='Fasttext')
    if sbert:
        ScoringEvaluation(dfEval, dfSBERT, topK, name='SBERT')
        top1000Evaluation(dfTop, dfSBERT, topK, name='SBERT')
