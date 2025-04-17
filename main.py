from datasets import load_dataset
from dataset import HingDataset
from load_embeddings import combine_embeddings_word2indices
from seq2seq import Encoder, Decoder, Seq2Seq

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import tqdm


# load HF hinglish to english parallel corpus
ds_train = load_dataset("festvox/cmu_hinglish_dog", split='train')

trans = ds_train["translation"]
en_examples = []
hi_examples = []

for t in trans:
    en_examples.append(t['en'])
    hi_examples.append(t['hi_en'])

w2index, embedding_layer = combine_embeddings_word2indices()
train_data = HingDataset(w2index, hi_examples, en_examples)

# The following few lines check whether a GPU is available, and if so,
# they run everything on a GPU which will be much faster.
device = "cpu"
if torch.cuda.is_available():
  device = "cuda"
print(f"Using {device} device")

v_size = w2index["<pad>"] + 1
h_size = 128
e_size = 300

lstm_encoder = Encoder(vocab_size = v_size, embedding_dimension = e_size, hidden_dimension = h_size)
lstm_encoder.to(device)
# We use a cross-entropy loss here
loss_fn = nn.CrossEntropyLoss().to(device)
# Finally, let's define the optimiser
optimizer = torch.optim.Adam(lstm_encoder.parameters(), lr=1e-3)

# This loads the data for training
train_dataloader = DataLoader(train_data, batch_size=40, shuffle=True)

# The function for running model training
def train(model, loss_fn, optimizer, dataloader, epochs=150):
    # number of times we go through the training data
    # set the mode of the model to training so that parameters are being updated
    model.train()

    for epoch in range(epochs):

        # intialize the loss at the beginning of each epoch
        epoch_losses = []

        # iterate through all training batches
        for X, y in tqdm.tqdm(dataloader):


            seq_len = X.size(1)
            batch_len = X.size(0)

            # move the input and the labels to the GPU, if we are using a GPU
            X = X.to(device)
            y = y.to(device).flatten()
            # y = y.to(device)

            # Initialise the hidden representation (this is h0)
            h = None

            # reset the optimiser
            optimizer.zero_grad()

            # make predictions using the model
            output, h = model(X, h)
            # print("Input shape:", X.shape)       # what goes into model
            # print("Output shape:", output.shape)     # what comes out
            # print("Target shape:", y.shape)     # your labels

            # compute the loss for the current predictions
            loss = loss_fn(output, y)
            # perform backpropagation
            loss.backward()
            # and update the weights
            optimizer.step()

            # add the loss of the current batch to the loss of the epoch
            epoch_losses.append(loss.item())

            if len(epoch_losses) % 1000 == 0:
                print("\nEpoch: {0}, current loss: {1}, ".format(epoch, sum(epoch_losses)/len(epoch_losses)))


        # print the loss at the end of every epoch
        print("\nEpoch: {0}, final loss: {1}, ".format(epoch, sum(epoch_losses)/len(epoch_losses)))

##steps to build model from scratch
# 1. get parallel corpus and clean - DONE
#
# 2. word-level language identification via HingBERT - DONE
#
# 3. split hinglish data up by sub-word tokens (based on language) - use HingBERT outputs
#
# 4. build a token dictionary (one for english, one for hindi)
#
# 5. convert input tokenized text into numbers corresponding to token dict --
#     - how to ensure language is preserved? should we create just one dictionary with all tokens in them
#
# 6. Model inputs:
#       - input text in both languages (??) converted from str to sequences of numbers
#       - vector embeddings for both languages
#
# 7. Model creation: will need to use some pretrained model
#       - need to look into how to set up encoder-decoder architecture
#       - guess: we set up one model that is the encoder and one that decodes the encoder's outputs into the target language
# 8. Fine-tune / train pretrained model
# 9. Try to generate outputs


# questions for Sebby:
#     - open question: how do we disambiguate for words that have matches in both the english and hindi embeddings?
#           - can we use context to disambiguate?
#     - do we even need POS tags?***
