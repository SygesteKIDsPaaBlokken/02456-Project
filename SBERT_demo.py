# %% Jupyter extensions
%load_ext autoreload
%autoreload 2
# %% Imports
from pathlib import Path
from tqdm import tqdm
import nltk
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import semantic_search
from utils.config import DEVICE

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
# %% Helper functions
def remove_stop_words(passages):
    global stop_words

    passages_clean = []
    for passage in tqdm(passages):  # ~20
        temp = ''
        for word in passage.split(' '):
            if word not in stop_words:
                temp += word + ' '
        passages_clean.append(temp[:-1])
    return passages_clean

# %% Model settings
models_path = Path('/dtu/blackhole/1b/167931/SBERT_models')
model_name = Path('1epoch')
SBERT_path = models_path / model_name
# %% Load model
SBERT = SentenceTransformer(SBERT_path, device=DEVICE)

# %% Demo
query = "What is Hygge?"
corpus = [
    "I think it's a short story. So I thought I might need more details. You're on Twitter and facebook",
    "Hygge is a Danish term that embodies a quality of coziness, comfort, and contentment."
]

# Cleaning
corpus_clean = remove_stop_words(corpus)
query_clean = remove_stop_words([query])

# Embed
corpus_embeddings = SBERT.encode(corpus_clean, device=DEVICE) 
query_embeddings = SBERT.encode(query_clean, device=DEVICE) 

# Search
query_results = semantic_search(query_embeddings, corpus_embeddings, top_k=10)[0]

# Visualize response
print(f"Query (Cleaned): {query_clean[0]}")
print("Results:")
for i, query_result in enumerate(query_results):
    print(f"{i+1} ({query_result['score']}): {corpus[query_result['corpus_id']]}")