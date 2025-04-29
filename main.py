from datasets import load_dataset
from dataset import HingDataset
# from load_embeddings import combine_embeddings_word2indices

from seq2seq import Encoder, Decoder, Seq2Seq
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import tqdm

ds_train = load_dataset("festvox/cmu_hinglish_dog", split='train')
ds_valid = load_dataset("festvox/cmu_hinglish_dog", split='validation')
ds_test = load_dataset("festvox/cmu_hinglish_dog", split='test')

train_trans = ds_train["translation"]
valid_trans = ds_valid["translation"]
test_trans = ds_test["translation"]

en_examples_train = [t['en'] for t in train_trans]
hi_examples_train = [t['hi_en'] for t in train_trans]
en_examples_valid = [t['en'] for t in valid_trans]
hi_examples_valid = [t['hi_en'] for t in valid_trans]
en_examples_test = [t['en'] for t in test_trans]
hi_examples_test = [t['hi_en'] for t in test_trans]

# w2index, embedding_layer = combine_embeddings_word2indices()

from torchtext import vocab

fasttext_hi = vocab.Vectors(name='wiki.hi.align.vec', url='https://dl.fbaipublicfiles.com/fasttext/vectors-aligned/wiki.hi.align.vec')
fasttext_en = vocab.Vectors(name='wiki.en.align.vec', url='https://dl.fbaipublicfiles.com/fasttext/vectors-aligned/wiki.en.align.vec')

train_data = HingDataset(hi_examples_train, en_examples_train)

# The following few lines check whether a GPU is available, and if so,
# they run everything on a GPU which will be much faster. Use if running in the cloud
# device = "cpu"
# if torch.cuda.is_available():
#   device = "cuda"
# print(f"Using {device} device")

# use this statement if running on machine
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

train_data = HingDataset(hi_examples_train, en_examples_train)
valid_data = HingDataset(hi_examples_valid, en_examples_valid)
test_data = HingDataset(hi_examples_test, en_examples_test)

hing_vocab = train_data.in_vocab
eng_vocab = train_data.out_vocab

input_size_encoder = len(hing_vocab)
input_size_decoder = len(eng_vocab)

h_size = 128
e_size = 300
n_layers = 2
enc_dropout = 0.2
dec_dropout = 0.2

lstm_encoder = Encoder(source_vocab_size=input_size_encoder, embedding_dimension=e_size, hidden_dimension=h_size, num_layers=n_layers, p=enc_dropout)
lstm_encoder.to(device)

lstm_decoder = Decoder(target_vocab_size=input_size_decoder, embedding_size=e_size, hidden_size=h_size, num_layers=n_layers, p=dec_dropout)
lstm_decoder.to(device)

model = Seq2Seq(lstm_encoder, lstm_decoder, device)
model.to(device)

# We use a cross-entropy loss here
loss_fn = nn.CrossEntropyLoss().to(device)
# Finally, let's define the optimiser
optimizer = torch.optim.Adam(lstm_encoder.parameters(), lr=1e-3)

# This loads the data for training
train_dataloader = DataLoader(train_data, batch_size=32, shuffle=True)
valid_dataloader = DataLoader(valid_data, batch_size=32, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=32, shuffle=True)


# The function for running model training
def train(model, loss_fn, optimizer, dataloader, epochs=100):
    # number of times we go through the training data
    # set the mode of the model to training so that parameters are being updated
    model.train()

    for epoch in range(epochs):
        print(f"[Epoch {epoch} / {epochs}]")

        # checkpoint = {"state_dict": model.state_dict(), "optimizer": optimizer.state_dict()}
        # save_checkpoint(checkpoint)

        # intialise the loss at the beginning of each epoch
        epoch_losses = []

        # iterate through all training batches
        for X, y in tqdm.tqdm(dataloader):

            seq_len = X.size(1)
            batch_len = X.size(0)

            # move the input and the labels to the GPU, if we are using a GPU
            X = X.to(device)
            y = y.to(device)  # .flatten()

            # Initialise the hidden representation (this is h0)
            h = None

            # reset the optimiser
            optimizer.zero_grad()

            # make predictions using the model
            output = model(X, y)

            output = output[1:].reshape(-1, output.shape[2])
            y = y[1:].reshape(-1)
            # compute the loss for the current predictions
            loss = loss_fn(output, y)
            # perform backpropagation
            loss.backward()

            # Clip to avoid exploding gradient issues, makes sure grads are
            # within a healthy range
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)

            # and update the weights
            optimizer.step()

            # add the loss of the current batch to the loss of the epoch
            epoch_losses.append(loss.item())

            if len(epoch_losses) % 1000 == 0:
                print("\nEpoch: {0}, current loss: {1}, ".format(epoch, sum(epoch_losses) / len(epoch_losses)))

            train_loss = sum(epoch_losses) / len(epoch_losses)

        # print the loss at the end of every epoch
        print("\nEpoch: {0}, final loss: {1}, ".format(epoch, train_loss))

train(model, loss_fn, optimizer, train_dataloader, epochs=2)

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


# questions:
#     - open question: how do we disambiguate for words that have matches in both the english and hindi embeddings?
#           - can we use context to disambiguate?
#     - do we even need POS tags?***
