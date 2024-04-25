from torch import nn
from transformers import T5ForConditionalGeneration, T5Tokenizer

class SuffixT5(nn.Module):
    def __init__(self, nb_suffix: int, base_model: nn.Module):
        super().__init__()
        self.base_model = base_model
        for param in self.base_model.parameters():
            param.requires_grad = False
        self.nb_suffix = nb_suffix
        self.suffix_embeddings = nn.Embedding(nb_suffix, base_model.encoder.embed_tokens.embedding_dim)
