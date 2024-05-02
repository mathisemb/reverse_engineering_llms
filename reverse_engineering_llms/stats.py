import torch
from transformers import T5ForConditionalGeneration
import matplotlib.pyplot as plt
import numpy as np

def word_std_wrt_dimension(matrix):
    # matrix is a V x D matrix where V is the vocab size and D the embedding size
    # returns a vector of size D containing the std of the words for each embedding dimension
    return torch.std(matrix, dim=0)

def word_norm(matrix):
    # matrix is a V x D matrix where V is the vocab size and D the embedding size
    # returns a vector of size D containing the L2 norm of the words for each embedding dimension
    return torch.norm(matrix, p=2, dim=0)

def distance_between_words(matrix):
    # matrix is a V x D matrix where V is the vocab size and D the embedding size
    # returns a vector of size V containing the L2 distance of each words from each word
    MMT = torch.matmul(matrix, matrix.t())
    word_norms = MMT.diag()
    TrMMT = word_norms.sum()
    TrMMT_tensor = torch.tensor([TrMMT]).repeat(matrix.shape[0])
    sums = torch.sum(MMT, dim=1)
    return word_norms - (2/matrix.shape[0])*sums + (1/matrix.shape[0])*TrMMT_tensor

def plot_array(array, sort, title, xlabel, ylabel):
    if sort:
        array = np.sort(array)
    plt.figure(figsize=(10, 6))
    plt.bar(range(array.shape[0]), array)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

if __name__ == "__main__":
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    model = T5ForConditionalGeneration.from_pretrained("t5-base")

    embed_matrix = model.encoder.embed_tokens.weight
    print("embed_matrix", embed_matrix.shape) # shpae = V x D

    stds = word_std_wrt_dimension(embed_matrix)
    plot_array(array = stds.detach().numpy(),
               sort = True,
               title = "Standard deviation of embedding dimensions",
               xlabel = "Dimension",
               ylabel = "Standard Deviation")

    norms = word_norm(embed_matrix)
    plot_array(array = norms.detach().numpy(),
               sort = True,
               title = "Norm of each word",
               xlabel = "Words",
               ylabel = "Norms")

    avg_distances = distance_between_words(embed_matrix)
    plot_array(array = avg_distances.detach().numpy(),
               sort = True,
               title = "Average distance of each word to each word",
               xlabel = "Words",
               ylabel = "Average distance to each word")
