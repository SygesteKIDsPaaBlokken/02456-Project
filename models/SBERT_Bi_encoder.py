#%% imports
from sentence_transformers import SentenceTransformer, models, InputExample, losses
from torch.utils.data import DataLoader
import torch
# %% Setup model
word_embedding_model = models.Transformer('nreimers/MiniLM-L6-H384-uncased', max_seq_length=512)
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
normalization_model = models.Normalize()
model = SentenceTransformer(modules=[word_embedding_model, pooling_model, normalization_model])

# Move model to GPU
use_cuda = True
device = 'cuda' if torch.cuda.is_available() and use_cuda else 'cpu'
model.to(device)
#%% Load training data
train_examples = [
    InputExample(texts=['Anchor 1', 'Positive 1', 'Negative 1']),
    InputExample(texts=['Anchor 2', 'Positive 2', 'Negative 2'])
    ]
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=32)

#%% Define loss
train_loss = losses.MultipleNegativesRankingLoss(model)

#%% Train model
model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=1, warmup_steps=100, show_progress_bar=True)

#%%