import torch

batch_size = 64
block_size = 256
device = 'cuda' if torch.cuda.is_available() else 'cpu'

with open('conversations.txt', mode='r', encoding='utf-8') as f:
    text = f.read()

def get_batch(data):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i: i + block_size] for i in ix])
    y = torch.stack([data[i + 1: i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y