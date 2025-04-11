# function to parse CONLLU files into sentences and labels
def read_conllu_file(file_path):
  sentences = []
  unique_labels = set()

  with open(file_path, "r", encoding="UTF-8") as in_f:
    current_sentence = []
    for line in in_f:
      line = line.strip()
      # ignore lines starting with # (comments)
      if line.startswith("#"):
        continue

      # an empty line indicates the end of the sentence
      if line == "":
        sentences.append(current_sentence)
        current_sentence = []
        continue

      # split the line into its parts
      parts = line.split("\t")

      # extract the index (the first column)
      idx = parts[0]

      # check if this is a multi-word token or an empty node
      if "." in idx or "-" in idx:
        continue

      if len(parts) < 4:
        print(parts)
      # extract the word and the tag, i.e., the second and fourth column
      word, tag = parts[1], parts[3]

      unique_labels.add(tag)

      # append the word, tag pair to the current sentence
      current_sentence.append((word, tag))

  return sentences, unique_labels