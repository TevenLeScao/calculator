import tokenizers
from os import path, listdir
prefix = path.join("../deepmind-gutenberg/train_concat")
files = [path.join(prefix, file_name) for file_name in listdir(prefix)]
tokenizer = tokenizers.ByteLevelBPETokenizer()
tokenizer.train(files, vocab_size=40000)
tokenizer.save_model("tokenizer_pg19")

