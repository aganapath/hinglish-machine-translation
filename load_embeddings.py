import torch
import torch.nn as nn
import numpy as np
import tqdm


def load_fasttext_embeddings(language):
    # download and extract the model for a given language, if it's not already downloaded
    NO_OF_DIMENSIONS = 300
    file_path = f"wiki.{language}.align.vec"
    word_to_index = {}

    with open(file_path, "r", encoding="UTF-8") as embedding_f:
        # this loops through the file and returns the line number in the file and the
        # contents of the line on each iteration
        for i, line in enumerate(embedding_f):
            # there is one line in the file that represents a space, which
            # can mess up the processing and therefore we ignore that line
            if i == 0 or line.startswith(" "):
                continue
            cols = line.split(" ")  # let's split the line into columns
            word = cols[0]  # the first column contains the word
            word_to_index[word] = i - 1  # add the word to the mapping

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
            cols = line.split(" ")  # let's split the line into columns
            idx = word_to_index[cols[0]]
            vector = np.array([float(c) for c in cols[1:]])  # get the word vector
            embeddings_np[idx, :] = vector
            if i > 500000:
                break

    return embeddings_np, word_to_index


def combine_embeddings_word2indices():
    embeddings_hi, hi_word_to_index = load_fasttext_embeddings('hi')
    print(embeddings_hi.shape)
    embeddings_en, en_word_to_index = load_fasttext_embeddings('en')
    print(embeddings_en.shape)

    embeddings_all = []
    embeddings_all = np.vstack([embeddings_hi, embeddings_en])

    # create a dictionary containing all words in Hindi and English in an index-to-word format
    # so as not to lost any words that exist in both Hi and En embeddings
    indices_to_word = {}
    for k, v in hi_word_to_index.items():
        indices_to_word[len(indices_to_word)] = k

    for k, v in en_word_to_index.items():
        indices_to_word[len(indices_to_word)] = k

    # add entries for <unk> and <pad> to the word_to_index dictionary
    indices_to_word[len(indices_to_word)] = "<unk>"
    indices_to_word[len(indices_to_word)] = "<pad>"

    # add the unk vector and the pad vector to the embedding matrix
    unk_vector = np.mean(embeddings_all, axis=0)
    pad_vector = np.zeros(unk_vector.shape)
    embeddings_all = np.vstack([embeddings_all, unk_vector, pad_vector])

    # convert the numpy embeddings to a torch tensor so that we can use
    # them in neural network model
    embeddings = torch.tensor(embeddings_all, dtype=torch.float32)

    # initialize an nn.Embedding object
    embedding_layer = nn.Embedding.from_pretrained(embeddings, freeze=True)

    return indices_to_word, embedding_layer

hing_indices_to_word, hing_embedding_layer = combine_embeddings_word2indices()