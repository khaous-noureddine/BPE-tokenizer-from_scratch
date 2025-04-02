import os
import time
from bpe_tokenizer import BPETokenizer

text = open("data/taylorswift.txt", "r", encoding="utf-8").read()
os.makedirs("models/napster_tokenizer", exist_ok=True)

start_time = time.time()
tokenizer = BPETokenizer()
vocab_size = 1000
# train the tokenizer :
tokenizer.train(text, vocab_size)
# save the tokenizer :
prefix = os.path.join("models/napster_tokenizer", "v1")
tokenizer.save(prefix)
end_time = time.time()
training_time = end_time - start_time
print(f"Training of tokenizer took {training_time:.2f} seconds for {vocab_size - 256} merges")