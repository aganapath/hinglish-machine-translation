import torch
from torch.utils.data import Dataset, DataLoader

class BertHingDataset(Dataset):
  def __init__(self, sequence, tokenizer, max_length = 512):
    self.sequence = sequence
    self.tokenizer = tokenizer
    self.max_length = max_length
    self.pad_token_id = tokenizer.pad_token_id #what does this do?
    self.pad_label_id = -100 #for

  def __getitem__(self, idx):

    # this is a shorthand for creating two lists
    # of tokens and tags for all the tuples in a
    # specific sentence
    # words, tags = zip(*self.sequence[idx])

    # tokens, tags = zip(*self.sentences[idx])
    # tokenized_sentences = []
    # sentences_lang_id = []

    tokenized_input = self.tokenizer(self.sequence,
                                    is_split_into_words=True,
                                    padding='max_length',
                                    truncation=True,
                                    max_length=self.max_length,
                                    return_tensors='pt')

    tokenized_seq = self.tokenizer.tokenize(self.sequence,
                                            is_split_into_words=True,
                                            padding='max_length',
                                            truncation=True,
                                            max_length=self.max_length,
                                            return_tensors='pt')

    # tokenized_sentences.append(tokenized_seq)

    input_ids = tokenized_input["input_ids"].squeeze(0)
    attention_mask = tokenized_input["attention_mask"].squeeze(0)
    word_ids = tokenized_input.word_ids()

    with torch.no_grad():
        logits = model(**tokenized_input).logits

    predicted_token_class_ids = logits.argmax(-1)
    predicted_tokens_classes = [model.config.id2label[t.item()] for t in predicted_token_class_ids[0]]

    sentence_lang_id = []

    prev_word_idx = None
    for i, word_idx in enumerate(word_ids):
        if word_idx is None:
            continue
        if word_idx != prev_word_idx:  # Only label the first subword of a token
            sentence_lang_id.append(predicted_tokens_classes[i])
        prev_word_idx = word_idx

    # sentences_lang_id.append(sentence_lang_id)

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask}
