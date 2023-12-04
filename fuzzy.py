import numpy as np
import pandas as pd
from tqdm import tqdm
from time import time
from fuzzysearch import find_near_matches
import nltk
from nltk.corpus import stopwords
from utils.evaluation import ScoringEvaluation


def rank_fuzzy(queries, queryIDs, passages, N=1000, max_l_dist=1, name='fuzzy'):
    try:
        out = pd.read_csv(f'data/{name}_RankingResults.csv', sep=',')
        return out
    except FileNotFoundError:
        pass
    queryColumn, passageColumn, scoreColumn = [np.empty(len(queries) * N) for _ in range(3)]

    i = 0
    for query in tqdm(queries, desc=name):  # ~3 minutes per query
        scores = []
        for passage in tqdm(passages):
            score = 0
            for word in query.split(' '):
                score += len(find_near_matches(word, passage, max_l_dist=max_l_dist))
            scores.append(score)

        scores = np.array(scores)
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

    df_fuzzy = rank_fuzzy(queries_clean, qids, passages_clean, max_l_dist=1, )
    print(f'Fuzzy searching took {(time() - t) / 60:.3f} minutes')  # ~

    topK = [1, 2, 3, 4, 5, 10, ]
    dfEval = pd.read_csv('data/0-3scoringTestSet.txt', sep=' ')
    ScoringEvaluation(dfEval, df_fuzzy, topK=topK, name='fuzzy')
    pass
