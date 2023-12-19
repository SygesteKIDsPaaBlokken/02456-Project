from enum import Enum

from utils.config import SBERT_MODELS_PATH

class MODEL(Enum):
    FUZZY_l1 = 'Fuzzy_l1'
    BM25 = 'BM25'
    FAST_TEXT = 'FastText'
    SBERT_1e = 'SBERT_1e'
    SBERT_2e = 'SBERT_2e'
    SBERT_3e = 'SBERT_3e'
    HP_SBERT = 'HP_SBERT'

def get_SBERT_model_path(model: MODEL):
    if model == MODEL.HP_SBERT:
        return SBERT_MODELS_PATH.joinpath('HP_model')
    
    if model == MODEL.SBERT_1e:
        return SBERT_MODELS_PATH.joinpath('1epoch')
    
    if model == MODEL.SBERT_2e:
        return SBERT_MODELS_PATH.joinpath('2epochs')
    
    if model == MODEL.SBERT_3e:
        return SBERT_MODELS_PATH.joinpath('3epochs')