import numpy as np
import pandas as pd
from tqdm import tqdm


def ScoringEvaluation(dfEval, dfModel, topK, name):
    results = [[] for _ in topK]
    resultsCount = [[] for _ in topK]
    resultsMax = [[] for _ in topK]

    query_ids = np.unique(dfEval['query'])
    for qid in tqdm(query_ids):
        
        # Find the relevant rows
        query_eval_rows = dfEval[dfEval['query'] == qid]
        for i, k in enumerate(topK):
            # Take the top k scores in the evaluation data, to find the maximum possible score
            resultsMax[i].append(
                np.sort(query_eval_rows['score'].values)[::-1][:k]\
                    .sum()
            )

            # Determine the top k summed score for the query
            passages = dfModel[dfModel['query'] == qid][:k]['passage']
            score = 0
            count = 0
            for passage in passages:
                temp = sum(query_eval_rows[query_eval_rows['passage'] == passage]['score'].values)
                score += temp
                count += 0 < temp

            results[i].append(score)
            resultsCount[i].append(count)

    print(name)
    sFT = ''
    score = []
    max_score = []
    count = []
    max_count = []
    for k, resultMax, resultFT, resultCount in zip(topK, resultsMax, results, resultsCount):
        score.append(sum(resultFT))
        max_score.append(sum(resultMax))
        count.append(sum(resultCount))
        max_count.append(k * len(query_ids))
    
    pd.DataFrame({'topK':topK,'score':score,'max_score':max_score,'count':count,'max_count':max_count}).to_csv('data/' + name + '.csv')

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
