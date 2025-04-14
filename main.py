from datasets import load_dataset
from conllu_data import read_conllu_file
from transformers import AutoTokenizer, AutoModelForTokenClassification
from dataset import BertHingDataset

ds_train = load_dataset("festvox/cmu_hinglish_dog", split='train')

trans = ds_train["translation"]
en_examples = []
hi_examples = []

for t in trans:
    en_examples.append(t['en'])
    hi_examples.append(t['hi_en'])

tokenizer = AutoTokenizer.from_pretrained("l3cube-pune/hing-bert-lid")
train_dataset = BertHingDataset(hi_examples, tokenizer, max_length=512)

model = AutoModelForTokenClassification.from_pretrained("l3cube-pune/hing-bert-lid")


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
