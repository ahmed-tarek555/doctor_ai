import torch
import pickle
from utils import device, block_size
from model import LanguageModel
from tokenizer import Tokenizer, vocab_size

tokenizer = Tokenizer(vocab_size)
model = LanguageModel(vocab_size)
model.load_state_dict(torch.load('parameters.pth', map_location=torch.device(device)))

with open('vocab.pkl', 'rb') as f:
    vocab = pickle.load(f)

idx = torch.zeros(1, 1, dtype=torch.long)
generated_tokens = model.generate(idx, 1000)[0][block_size:].tolist()
print(tokenizer.decode(generated_tokens, vocab))