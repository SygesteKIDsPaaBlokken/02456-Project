from time import time
from fuzzysearch import find_near_matches
from pandas import DataFrame
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm

from utils import MODEL, get_SBERT_model_path

from utils.config import DEVICE, MODEL_NAME, RANKING_PATH, VERBOSE
from utils.preprocessing import remove_stop_words

def _save_rankings(rankings: DataFrame, save_path: str):
    if VERBOSE:
        t = time()
        print('# Saving rankings')

        rankings.to_csv(save_path, index=False)
    
    if VERBOSE:
        print(f'# Rankings saved: {(time() - t)/60:.3f} min')

def rank_sbert(
        queries: DataFrame,
        passages: DataFrame,
        model_path: str = get_SBERT_model_path(MODEL.HP_SBERT),
        top_k: int = 100,
        model_name: str = MODEL_NAME,
        save_rankings: bool = True,
        save_path: str = None
        ):
    save_rankings = save_rankings or save_path is not None
    if save_path is None: save_path = RANKING_PATH.joinpath(f'{model_name}_top{top_k}.csv')

    if VERBOSE:
        print("Using device:", DEVICE)
        print("Evaluating model:", model_path)

    model = SentenceTransformer(model_path, device=DEVICE)
    
    if VERBOSE:
        t = time()
        print('# Encoding corpus')
    SBERTCorpus = model.encode(passages['passage'].to_numpy(), batch_size=32, show_progress_bar=True, device=DEVICE) 
    if VERBOSE:
        print(f'# Corpus encoding finished: {(time() - t)/60:.3f} min')

    if VERBOSE:
        t = time()
        print('# Encoding queries')
    query_embeddings = model.encode(queries['query'].to_numpy(), batch_size=32, show_progress_bar=True, device=DEVICE)
    if VERBOSE:
        print(f'# Queries encoding finished: {(time() - t)/60:.3f} min')
    
    if VERBOSE:
        t = time()
        print('# Ranking passages')
    ranking = util.semantic_search(query_embeddings, SBERTCorpus, top_k=top_k)
    if VERBOSE:
        print(f'# Ranking finished: {(time() - t)/60:.3f} min')
    
    # Transform into known query and passage ids
    if VERBOSE:
        t = time()
        print('# Transforming rankings')
    transformed_rankings = []
    for query_idx, query_results in enumerate(tqdm(ranking, desc='Transforming',)):
        qid = queries.iloc[query_idx]['qid']
        
        for result in query_results:
            transformed_rankings.append({
                'qid': qid,
                'pid': passages.iloc[result['corpus_id']]['pid'],
                'score': result['score']
            })
        
    transformed_rankings = DataFrame(transformed_rankings)
    if VERBOSE:
        print(f'# Ranking transformation finished: {(time() - t)/60:.3f} min')
    
    if save_rankings:
        _save_rankings(transformed_rankings, save_path)
    
    return transformed_rankings

def rank_fuzzy(
        queries: DataFrame,
        passages: DataFrame,
        top_k: int = 100,
        max_l_dist = 1,
        save_rankings: bool = True,
        save_path: str = None,
        rm_stop_words: bool = True
    ):

    save_rankings = save_rankings or save_path is not None
    if save_path is None: save_path = RANKING_PATH.joinpath(f'Fuzzy_l{max_l_dist}_top{top_k}.csv')

    if rm_stop_words:
        if VERBOSE:
            t = time()
            print('# Removing stop words')
        passages['passage'] = remove_stop_words(passages['passage'])
        queries['query'] = remove_stop_words(queries['query'])
        if VERBOSE:
            t = time()
            print('# Removing stop words')

    if VERBOSE:
        print(f'# Stop words removed: {(time() - t)/60:.3f} min')

    rankings = []
    for i, query_row in tqdm(queries.iterrows(),total=len(queries)):
        qid, query = query_row['qid'], query_row['query']
        words_in_query = query.split()

        passage_scores = passages['passage'].apply(
            lambda p: sum([len(find_near_matches(word, p, max_l_dist=max_l_dist)) for word in words_in_query])\
                /(len(words_in_query)+len(p))
        )

        top_k_sorting = (-passage_scores).argsort()[:top_k]
        top_k_passages = passages.loc[top_k_sorting, 'pid']
        top_k_scores = passage_scores.loc[top_k_sorting]

        rankings.extend(
            [{'qid':qid, 'pid':pid, 'score': score} for pid, score in zip(top_k_passages,top_k_scores)]
        )

    rankings = DataFrame(rankings)
    if VERBOSE:
        print(f'# Ranking finished: {(time() - t)/60:.3f} min')
    
    if save_rankings:
        _save_rankings(rankings, save_path)
    
    return rankings