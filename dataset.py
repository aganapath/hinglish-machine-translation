import torch
from torch.utils.data import Dataset

# create a dataset class that will do the following:
# 1. load the text file and tokenize the sequences of words
# 2. make the output of the tokenized inputs consistent with the input words (in case they were split up during tokenization)
# 3. create a label2index thingy that is consistent with the labels of the original words
# 4. lastly generate the tensors for inputs and labels that will get passed into the model
# BERT model wants 1) input_ids - sequence of indices that correspond to each word's POS tag
#                  2) attention_mask - will output from tokenizer
#                  3) labels - sequence (list) of

class HingDataset(Dataset):
  def __init__(self, word_to_indices, in_sequences, out_sequences, max_length = 30):
    self.word_to_indices = word_to_indices
    self.in_sequences = in_sequences
    self.out_sequences = out_sequences
    # self.tokenizer = tokenizer
    # self.label2index = label2index
    self.max_length = max_length
    # self.pad_token_id = tokenizer.pad_token_id #what does this do?
    # self.pad_label_id = -100 #for

  def __len__(self):
    return len(self.in_sequences)

  def __getitem__(self, idx):
    input_tokens = []
    self.in_sequence = self.in_sequences[idx].lower().split()
    self.out_sequence = self.out_sequences[idx].lower().split()

    for w in self.in_sequence:
      w_str = w.strip(',')
      if w_str in self.word_to_indices['hi']:
        input_tokens.append(self.word_to_indices['hi'][w_str])
      elif w_str in self.word_to_indices['en']:
        input_tokens.append(self.word_to_indices['en'][w_str])
      else:
        input_tokens.append(self.word_to_indices['<unk>'])

    output_tokens = []
    for w in self.out_sequence:
      w_str = w.strip('.,!?:\'').replace('\'', '')
      if w_str in self.word_to_indices['en']:
        output_tokens.append(self.word_to_indices['en'][w_str])
      else:
        output_tokens.append(self.word_to_indices['<unk>'])

    # pad and truncate
    if len(input_tokens) < self.max_length:
      diff = self.max_length - len(input_tokens)
      padding = [self.word_to_indices['<pad>']] * diff
      input_tokens = padding + input_tokens

    input_tokens = input_tokens[-self.max_length:]

    if len(output_tokens) < self.max_length:
      diff = self.max_length - len(output_tokens)
      padding = [self.word_to_indices['<pad>']] * diff
      output_tokens = padding + output_tokens

    output_tokens = output_tokens[-self.max_length:]

    input_ids = torch.LongTensor(input_tokens)
    output_ids = torch.LongTensor(output_tokens)

    return input_ids, output_ids
