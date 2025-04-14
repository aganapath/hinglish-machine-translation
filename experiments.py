# read in pacman sample
# with open('PACMAN_sample_1000.txt', 'r', encoding='utf8') as file:
#     txt_cleaned = []
#     for line in file:
#         l = []
#         phrase = line.split()
#         for w in phrase:
#             word = w.split('\\')
#             l.append(word[0])
#
#         txt_cleaned.append(l)

# experimenting with tokenizer
# from transformers import AutoTokenizer

# tokenizer = AutoTokenizer.from_pretrained("obaidtambo/hinglish_bert_tokenizer")
#
# example = "you're such a liar"
# tokens = tokenizer.tokenize(example)
# print(tokens)


# pulling in Hindi conllu data
# train_data_sentences, unique_labels = read_conllu_file("hi_hdtb-ud-train.conllu")

# print(unique_labels)
#
# for i in range(1):
#     t = train_data_sentences[i]
#     for tup in t:
#         hi_word = tup[0]
#         for h in hi_word:
#             print(h)
#
#     print(train_data_sentences[i])