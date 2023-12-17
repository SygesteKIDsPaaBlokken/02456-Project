from time import time
from pandas import DataFrame
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm
from utils import MODEL, get_SBERT_model_path

from utils.config import DEVICE, MODEL_NAME, VERBOSE

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
    if save_path is None: save_path = f'data/rankings/{model_name}_{top_k}.csv'

    if VERBOSE:
        print("Using device:", DEVICE)
        print("Evaluating model:", model_path)

    model = SentenceTransformer(model_path, device=DEVICE)
    
    if VERBOSE:
        t = time()
        print('########## Encoding corpus ##########')
    SBERTCorpus = model.encode(passages['passage'].to_numpy(), batch_size=32, show_progress_bar=True, device=DEVICE) 
    if VERBOSE:
        print(f'########## Corpus encoding finished ({(time() - t)/60:.3f} min) ##########')

    if VERBOSE:
        t = time()
        print('########## Encoding queries ##########')
    query_embeddings = model.encode(queries['query'].to_numpy(), batch_size=32, show_progress_bar=True, device=DEVICE)
    if VERBOSE:
        print(f'########## Queries encoding finished ({(time() - t)/60:.3f} min) ##########')
    
    if VERBOSE:
        t = time()
        print('########## Ranking passages ##########')
    ranking = util.semantic_search(query_embeddings, SBERTCorpus, top_k=top_k)
    if VERBOSE:
        print(f'########## Ranking finished ({(time() - t)/60:.3f} min) ##########')
    
    # Transform into known query and passage ids
    if VERBOSE:
        t = time()
        print('########## Transforming rankings ##########')
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
        print(f'########## Ranking transformation finished ({(time() - t)/60:.3f} min) ##########')
    
    if save_rankings:
        if VERBOSE:
            t = time()
            print('########## Saving rankings ##########')
        transformed_rankings.to_csv(save_path, index=False)
        if VERBOSE:
            print(f'########## Rankings saved ({(time() - t)/60:.3f} min) ##########')
    
    return transformed_rankings