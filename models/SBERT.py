from sentence_transformers import SentenceTransformer, models

class SBERT:
    def __init__(self,
                 word_embedding_model_name: str = 'nreimers/MiniLM-L6-H384-uncased',
                 max_seq_length: int = 384,
                 pooling_mode: str = None,
                 device: str = None):
        word_embedding_model = models.Transformer(word_embedding_model_name, max_seq_length=max_seq_length)
        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), pooling_mode=pooling_mode)
        normalization_model = models.Normalize()
        self.model = SentenceTransformer(
            modules=[word_embedding_model,pooling_model,normalization_model],
            device=device
        )