from datasets import load_dataset
from dataset import HingDataset
# from load_embeddings import combine_embeddings_word2indices
from seq2seq import Encoder, Decoder, Seq2Seq
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from train_eval import train, evaluate


# The following few lines check whether a GPU is available, and if so,
# they run everything on a GPU which will be much faster. Use if running in the cloud
# device = "cpu"
# if torch.cuda.is_available():
#   device = "cuda"
# print(f"Using {device} device")

# use this statement if running on machine
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


hing_ds = load_dataset("findnitai/english-to-hinglish")
hing_ds = hing_ds['train']
hing_ds_split = hing_ds.train_test_split(test_size=0.05)

train_trans = hing_ds_split["train"]
test_trans = hing_ds_split["test"]

hi_en_train_trans = train_trans["translation"]
hi_en_test_trans = test_trans["translation"]

en_examples_train = [t['en'] for t in hi_en_train_trans]
hi_examples_train = [t['hi_ng'] for t in hi_en_train_trans]

en_examples_test = [t['en'] for t in hi_en_test_trans]
hi_examples_test = [t['hi_ng'] for t in hi_en_test_trans]

train_data = HingDataset(hi_examples_train, en_examples_train)
test_data = HingDataset(hi_examples_test, en_examples_test)

hing_vocab = train_data.in_vocab
eng_vocab = train_data.out_vocab
sos_idx = eng_vocab.get_stoi()["<sos>"]



# loads embeddings for hindi and english - may not need to use this
# fasttext_hi = vocab.Vectors(name='wiki.hi.align.vec', url='https://dl.fbaipublicfiles.com/fasttext/vectors-aligned/wiki.hi.align.vec')
# fasttext_en = vocab.Vectors(name='wiki.en.align.vec', url='https://dl.fbaipublicfiles.com/fasttext/vectors-aligned/wiki.en.align.vec')

input_size_encoder = len(hing_vocab)
input_size_decoder = len(eng_vocab)

h_size = 512
e_size = 300
b_size = 128
n_layers = 2
enc_dropout = 0.2
dec_dropout = 0.2

# This loads the data for training
# potentially debug this
train_dataloader = DataLoader(train_data, batch_size=b_size, shuffle=False)
test_dataloader = DataLoader(test_data, batch_size=b_size, shuffle=False)

lstm_encoder = Encoder(source_vocab_size=input_size_encoder, embedding_dimension=e_size, hidden_dimension=h_size, num_layers=n_layers, p=enc_dropout)
lstm_encoder.to(device)

lstm_decoder = Decoder(target_vocab_size=input_size_decoder, embedding_size=e_size, hidden_size=h_size, num_layers=n_layers, p=dec_dropout, device=device)
lstm_decoder.to(device)

model = Seq2Seq(lstm_encoder, lstm_decoder, device)
model.to(device)

# We use a cross-entropy loss here
loss_fn = nn.CrossEntropyLoss().to(device)
# Finally, let's define the optimiser
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


train(model, loss_fn, optimizer, train_dataloader, device, epochs=50)
torch.save(model.state_dict(), '/Users/anjaniganapathy/PycharmProjects/hinglish-machine-translation/hinglish_model.pt')

evaluate(model, test_dataloader, eng_vocab, device, sos_idx)


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