from datasets import load_dataset

ds_train = load_dataset("festvox/cmu_hinglish_dog", split='train')

trans = ds_train["translation"]
print(trans)
