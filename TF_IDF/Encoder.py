from collections import defaultdict
import gc
import numpy as np
from numpy.linalg import norm
from pandas import DataFrame
from tqdm import tqdm
from nltk.corpus import stopwords
from scipy.spatial.distance import cdist
from sklearn.metrics.pairwise import cosine_similarity

from Loader import clean_text

def cosine_similarity(A: np.ndarray, B: np.ndarray) -> float:
    return np.dot(A, B)/(norm(A)*norm(B))

class TFIDF:
    def __init__(self,
                 documents: DataFrame,
                 stop_words: list[str] = set(stopwords.words('english')),
                 chunk_size: int = 10_000,
                 chunk_cache: str = '/dtu/blackhole/1b/167931/tf_idf/chunks') -> None:
        document_dict = {pid: clean_text(passage) for pid, passage in zip(documents['pid'],documents['passage'])}
        self.stop_words = stop_words
        self.chunk_size = chunk_size
        self.chunk_cache = chunk_cache

        print('[Fitting] Computing term frequency')
        self.tf = {}
        self.idf = defaultdict(lambda: 0)
        for pid, doc in tqdm(document_dict.items()):
            doc_tf = defaultdict(lambda: 0)
            
            for term in doc.split():
                if term not in stop_words:
                    doc_tf[term] += 1
            
            if len(doc_tf.keys()) == 0: continue
            
            max_freq = max(doc_tf.values())
            for term in doc_tf.keys():
                doc_tf[term] = doc_tf[term]/max_freq
                self.idf[term] += 1

            self.tf[pid] = doc_tf
        
        print('[Fitting] Computing inverse document frequency')
        N = len(documents)
        for term in self.idf.keys():
            self.idf[term] = np.log2(N/self.idf[term])

        self.vocabulary = list(self.idf.keys())
        
        print('[Fitting] Computing feature vectors')
        chunk = np.zeros((chunk_size, len(self.vocabulary)))
        chunk_idx = 0
        for pid in tqdm(documents['pid']):
            doc_tf = self.tf[pid]
            doc_chunk_idx = pid % chunk_size
            
            for term in doc_tf.keys():
                term_idx = self.vocabulary.index(term)
                chunk[doc_chunk_idx, term_idx] = doc_tf[term]*self.idf[term]

            if pid % chunk_size == 0:
                np.save(f'{chunk_cache}/chunk_{chunk_idx}.npy', chunk)

                del chunk
                gc.collect()

                chunk = np.zeros((chunk_size, len(self.vocabulary)))
                chunk_idx += 1
        
        if pid % chunk_size > 0:
            np.save(f'{chunk_cache}/chunk_{chunk_idx}.npy', chunk)
        
        self.chunks = chunk_idx+1

        print('[Fitting] Completed')

    def search(self, query: str, number_of_results: int = 5):
        """Searches given corpus for best matches.
        
        Returns: Dict[int,float] = Dict[document id/pid, cosine similarity to query]
        """
        q = clean_text(query)

        # Compute term frequencies
        tf = self.compute_tf(q)
        # Create feature vector
        vec = self.compute_feature_vector(tf)
        # Search corpus
        results = {i:0 for i in range(number_of_results)}
        current_min = 0
        current_min_key = 0

        for chunk_idx in range(self.chunks):
            chunk = np.load(f'{self.chunk_cache}/chunk_{chunk_idx}.npy')

            similarities = cosine_similarity(chunk, vec)
        # for pid in tqdm(self.tf.keys()):
        #     doc_vec = self.compute_feature_vector(self.tf[pid])
        #     similarity = cdist(vec, doc_vec)
            
        #     if similarity > current_min:
        #         del results[current_min_key]
        #         results[pid] = similarity
        #         current_min_key = min(results, key=results.get)
        #         current_min = results[current_min_key]

        return similarities

    def compute_tf(self, text: str):
        tf = defaultdict(lambda: 0)
        for term in text.split():
            if term not in self.stop_words:
                tf[term] += 1
        
        max_freq = max(tf.values())
        for term in tf.keys():
            tf[term] = tf[term]/max_freq

        return tf
    
    def compute_feature_vector(self, tf:dict):
        doc_feature_vector = np.zeros(len(self.vocabulary))  

        for term in tf.keys():
            idx = self.vocabulary.index(term)
            doc_feature_vector[idx] = tf[term]*self.idf[term]

        return doc_feature_vector