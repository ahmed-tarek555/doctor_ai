import pickle
import torch
from utils import device
from model import LanguageModel
from tokenizer import tokenizer

lr = 3e-4
model = LanguageModel(tokenizer.n_vocab)
model = model.to(device)
model = torch.compile(model)

with open('tokenized_data.pkl', 'rb') as f:
    tokens = pickle.load(f)

n = int(len(tokens) * 0.9)
train_tokens = tokens[:n]
train_tokens = torch.tensor(train_tokens).to(device)
model._train(train_tokens, 100, lr)

torch.save(model.state_dict(), 'parameters.pth')


