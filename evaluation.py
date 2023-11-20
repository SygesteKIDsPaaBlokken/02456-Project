import numpy as np
import pandas as pd
from tqdm import tqdm

def ScoringEvaluation(dfEval, dfModel, topK, name):
    results = [[] for _ in topK]
    resultsMax = [[] for _ in topK]

    for qid in tqdm(np.unique(dfEval['query'])):
        # Find the relevant columns to speed up computation:
        evalRows = dfEval[dfEval['query'] == qid]
        for i, k in enumerate(topK):
            # Take the top k scores in the evaluation data, to find the maximum possible score:
            resultsMax[i].append(np.sum(np.sort(evalRows['score'].values)[::-1][:k]))
            # Could be a list comprehension:
            passages = dfModel[dfModel['query'] == qid][:k]['passage']
            score = 0
            for passage in passages:
                score += sum(evalRows[evalRows['passage'] == passage]['score'].values)
            results[i].append(score)

    print(name)
    sFT = ''
    for k, resultMax, resultFT in zip(topK, resultsMax, results):
        sFT += f'{k}: {sum(resultFT)}/{sum(resultMax)}\t'
    print(sFT)


def top1000Evaluation(dfTop, dfModel, topK, name):
    results = [[] for _ in topK]
    for qid in tqdm(np.unique(dfTop.values[:, 0])):
        # Find the relevant columns to speed up computation:
        evalRows = dfTop[dfTop.values[:, 0] == qid].values[:, :2]
        for i, k in enumerate(topK):
            # Could be a list comprehension:
            passagesFT = dfModel[dfModel['query'] == qid][:k]['passage']

            results[i].append(sum([sum(passage == evalRows[:, 1]) for passage in passagesFT]))

    print(f'{name} (average overlap)')
    sFT = ''
    for k, resultFT in zip(topK, results):
        sFT += f'{k}: {np.average(resultFT):.2}/{k}\t'
    print(sFT)


if __name__ == '__main__':
    dfEval = pd.read_csv('0-3scoringTestSet.txt', sep=' ')
    dfTop = pd.read_csv('msmarco-passagetest2019-top1000.tsv', sep='\t', header=None)
    dfFT = pd.read_csv('Fasttext/RankingResults_0_200.csv', sep=',')

    topK = [1, 5, 10, 100, 1000]
    ScoringEvaluation(dfEval, dfFT, topK, name='Fasttext')
    top1000Evaluation(dfTop, dfFT, topK, name='Fasttext')

