import torch
import torch.nn as nn
import numpy as np
import tqdm


def load_fasttext_embeddings(languages):
    # intialize the dictionary for storing the embedding matrices
    embedding_matrices = {}
    # intialize the dictionary for storing the word2index dictionaries
    # for each language
    word2indices = {}
    for lang in languages:
        # download and extract the model for a given language, if it's not already downloaded
        NO_OF_DIMENSIONS = 300
        file_path = f"wiki.{lang}.align.vec"
        word_to_index = {}

        with open(file_path, "r", encoding="UTF-8") as embedding_f:
        # this loops through the file and returns the line number in the file and the
        # contents of the line on each iteration
            for i, line in enumerate(embedding_f):
                # there is one line in the file that represents a space, which
                # can mess up the processing and therefore we ignore that line
                if i == 0 or line.startswith(" "):
                    continue
                cols = line.split(" ") # let's split the line into columns
                word = cols[0] # the first column contains the word
                word_to_index[word] = i - 1 # add the word to the mapping

                if i > 500000:
                    break

        embeddings_np = np.zeros((len(word_to_index), NO_OF_DIMENSIONS))
        with open(file_path, "r", encoding="UTF-8") as embedding_f:
        # this loops through the file and returns the line number in the file and the
        # contents of the line on each iteration
            for i, line in tqdm.tqdm(enumerate(embedding_f)):
                # there is one line in the file that represents a space, which
                # can mess up the processing and therefore we ignore that line
                if i == 0 or line.startswith(" "):
                    continue
                cols = line.split(" ") # let's split the line into columns
                idx = word_to_index[cols[0]]
                vector = np.array([float(c) for c in cols[1:]]) # get the word vector
                embeddings_np[idx, :] = vector
                if i > 500000:
                    break


       # embeddings_np = np.loadtxt(file_path, skiprows=1, encoding="UTF-8", usecols=cols_to_use, comments=None)


        # compute a vector for unknown words and the padding token
        unk_vector = np.mean(embeddings_np, axis=0)
        pad_vector = np.zeros(unk_vector.shape)

        # add the unk vector and the pad vector to the embedding matrix
        embeddings_np = np.vstack([embeddings_np, unk_vector, pad_vector])

        # add entries for <unk> and <pad> to the word_to_index dictionary
        word_to_index["<unk>"] = len(word_to_index)
        word_to_index["<pad>"] = len(word_to_index)

        # convert the numpy embeddings to a torch tensor so that we can use
        # them in neural network model
        embeddings = torch.tensor(embeddings_np, dtype=torch.float32)
        # initialize an nn.Embedding object
        embedding_layer = nn.Embedding.from_pretrained(embeddings, freeze=True)

        # add both the Embedding object and the word_to_index dictionary to the
        # dictionary storing all the embedding matrices and word to index mappings
        embedding_matrices[lang] = embedding_layer
        word2indices[lang] = word_to_index
    return embedding_matrices, word2indices

languages = ['en', 'hi']
embedding_matrices, word2indices = load_fasttext_embeddings(languages)

print("Loaded embeddings for " + ", ".join(languages))