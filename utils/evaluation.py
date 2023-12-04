import numpy as np
import pandas as pd
from tqdm import tqdm


def ScoringEvaluation(dfEval, dfModel, topK, name):
    results = [[] for _ in topK]
    resultsCount = [[] for _ in topK]
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
            count = 0
            for passage in passages:
                temp = sum(evalRows[evalRows['passage'] == passage]['score'].values)
                score += temp
                count += 0 < temp
            results[i].append(score)
            resultsCount[i].append(count)

    print(name)
    sFT = ''
    for k, resultMax, resultFT, resultCount in zip(topK, resultsMax, results, resultsCount):
        sFT += f'Score {k}: {sum(resultFT)}/{sum(resultMax)}\t'
        sFT += f'Count {k}: {sum(resultCount)}/{k * len(np.unique(dfEval["query"]))}\n'
    print(sFT)
    with open('data/' + name + '.txt', 'w') as f:
        f.write(sFT)
    return results


def top1000Evaluation(dfTop, dfModel, topK, name):
    # USELESS
    results = [[] for _ in topK]
    for qid in tqdm(np.unique(dfTop.values[:, 0])):
        # Find the relevant columns to speed up computation:
        evalRows = dfTop[dfTop.values[:, 0] == qid].values[:, :2]
        for i, k in enumerate(topK):
            # Could be a list comprehension:
            passagesFT = dfModel[dfModel['query'] == qid][:k]['passage']
            # Check for each passage whether it is there
            results[i].append(sum([sum(passage == evalRows[:, 1]) for passage in passagesFT]))

    print(f'{name} (average overlap)')
    sFT = ''
    for k, resultFT in zip(topK, results):
        sFT += f'{k}: {np.average(resultFT):.2}/{k}\t'
    print(sFT)
    return results
