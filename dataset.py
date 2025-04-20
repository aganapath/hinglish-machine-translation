import torch
from torch.utils.data import Dataset, DataLoader
import torchtext

# create a dataset class that will do the following:
# 1. load the text file and tokenize the sequences of words
# 2. make the output of the tokenized inputs consistent with the input words (in case they were split up during tokenization)
# 3. create a label2index thingy that is consistent with the labels of the original words
# 4. lastly generate the tensors for inputs and labels that will get passed into the model
# BERT model wants 1) input_ids - sequence of indices that correspond to each word's POS tag
#                  2) attention_mask - will output from tokenizer
#                  3) labels - sequence (list) of


# potential improvement: use SpaCy tokenizer?
# potential improvement/concern: should we build word_to_indices on just the training data? It is important to note that a vocabulary should only
#                                 be built from the training set and never the validation or test set.
#                                 This prevents "information leakage" into our model, giving us artifically inflated validation/test scores.
#                                     source: https://github.com/bentrevett/pytorch-seq2seq/blob/main/1%20-%20Sequence%20to%20Sequence%20Learning%20with%20Neural%20Networks.ipynb

class HingDataset(Dataset):
  def __init__(self, in_sequences, out_sequences, max_length = 30):
    self.in_sequences = in_sequences
    self.out_sequences = out_sequences
    self.max_length = max_length
    self.in_vocab = self.create_vocab(self.in_sequences)
    self.out_vocab = self.create_vocab(self.out_sequences)

  def __len__(self):
    return len(self.in_sequences)

  def create_vocab(self, sequences):
    # create list of tokenized sentences to pass in to vocab creator
    sequences_tokenized = []
    for seq in sequences:
      seq_tokenized = []
      seq = seq.lower().split()
      for w in seq:
        w_str = w.strip('.,!?:\'').replace('\'', '')
        seq_tokenized.append(w)

      sequences_tokenized.append(seq_tokenized)

    # define special tokens
    unk_token = "<unk>"
    pad_token = "<pad>"
    sos_token = "<sos>"
    eos_token = "<eos>"

    special_tokens = [
        unk_token,
        pad_token,
        sos_token,
        eos_token,
    ]

    # create torchtext vocabulary object from sequences
    vocab = torchtext.vocab.build_vocab_from_iterator(
                      sequences_tokenized,
                      min_freq=2,
                      specials=special_tokens
                    )

    return vocab

  def __getitem__(self, idx):
    self.in_sequence = self.in_sequences[idx].lower().split()
    self.out_sequence = self.out_sequences[idx].lower().split()

    input_tokens = []
    for w in self.in_sequence:
      w_str = w.strip('.,!?:\'').replace('\'', '')
      if w_str in self.in_vocab:
        input_tokens.append(self.in_vocab.get_stoi()[w_str])
      else:
        input_tokens.append(self.in_vocab.get_stoi()['<unk>'])

    output_tokens = []
    for w in self.out_sequence:
      w_str = w.strip('.,!?:\'').replace('\'', '')
      if w_str in self.out_vocab:
        output_tokens.append(self.out_vocab.get_stoi()[w_str])
      else:
        output_tokens.append(self.out_vocab.get_stoi()['<unk>'])

    # pad and truncate
    if len(input_tokens) < self.max_length:
      diff = self.max_length - len(input_tokens)
      padding = [self.in_vocab.get_stoi()['<pad>']] * diff
      input_tokens = padding + input_tokens

    input_tokens = input_tokens[-self.max_length:]

    if len(output_tokens) < self.max_length:
      diff = self.max_length - len(output_tokens)
      padding = [self.out_vocab.get_stoi()['<pad>']] * diff
      output_tokens = padding + output_tokens

    output_tokens = output_tokens[-self.max_length:]

    input_ids = torch.LongTensor(input_tokens)
    output_ids = torch.LongTensor(output_tokens)

    return input_ids, output_ids
