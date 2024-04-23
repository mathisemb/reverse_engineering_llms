from torch import nn
from transformers import T5ForConditionalGeneration, T5Tokenizer

class SuffixT5(T5ForConditionalGeneration):
    def __init__(self, nb_suffix: int, model: nn.Module):
        super().__init__(model.config)
        self.nb_suffix = nb_suffix
        self.suffix_embeddings = nn.Embedding(nb_suffix, model.encoder.embed_tokens.embedding_dim)
