from collections import defaultdict
import numpy as np
from numpy.linalg import norm
from pandas import DataFrame
from tqdm import tqdm
from nltk.corpus import stopwords
from scipy.spatial.distance import cdist

from Loader import clean_text

def cosine_similarity(A: np.ndarray, B: np.ndarray) -> float:
    return np.dot(A, B)/(norm(A)*norm(B))

class TFIDF:
    def __init__(self,
                 documents: DataFrame,
                 stop_words: list[str] = set(stopwords.words('english'))) -> None:
        document_dict = {pid: clean_text(passage) for pid, passage in zip(documents['pid'],documents['passage'])}
        self.stop_words = stop_words

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
        
        # print('[Fitting] Computing feature vectors')
        # self.tf_idf = {}
        # for pid, doc_tf in tqdm(tf_dict.items()):
        #     doc_feature_vector = np.zeros(len(self.vocabulary))
            
        #     for i, term in enumerate(self.vocabulary):
        #         doc_feature_vector[i] = doc_tf[term]*self.idf[term]

        #     self.tf_idf[pid] = doc_feature_vector
        
        # print('[Fitting] Completed')
        # del tf_dict
        # gc.collect()

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
        for pid in tqdm(self.tf.keys()):
            doc_vec = self.compute_feature_vector(self.tf[pid])
            similarity = cdist(vec, doc_vec)
            
            if similarity > current_min:
                del results[current_min_key]
                results[pid] = similarity
                current_min_key = min(results, key=results.get)
                current_min = results[current_min_key]

        return results

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