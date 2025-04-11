from datasets import load_dataset
# from conllu_data import read_conllu_file

#
# train_data_sentences, unique_labels = read_conllu_file("data/en_ewt-ud-train.conllu")

ds_train = load_dataset("festvox/cmu_hinglish_dog", split='train')

trans = ds_train["translation"]
en_examples = []
hi_examples = []

for t in trans:
    en_examples.append(t['en'])
    hi_examples.append(t['hi_en'])

char2idx = {}

for e in en_examples:
    for char in e:
        if char not in char2idx:
            char2idx[char] = len(char2idx)


# printing equivalent hinglish/english sentences
# for i in range(5):
#     print(en_examples[i])
#     print(hi_examples[i])
# print(trans)

# experimenting with tokenizer
# from transformers import AutoTokenizer

# tokenizer = AutoTokenizer.from_pretrained("obaidtambo/hinglish_bert_tokenizer")
#
# example = "you're such a liar"
# tokens = tokenizer.tokenize(example)
# print(tokens)