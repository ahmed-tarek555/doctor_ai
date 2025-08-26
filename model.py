import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
from tokenizer import vocab_size
from utils import get_batch, block_size, device

n_emb = 384
n_heads = 6
dropout = 0.2

with torch.no_grad():
    def get_loss(data, model):
        model.eval()
        sum = 0
        len = 0
        for i in range(100):
            x, y = get_batch(data)
            loss = model(x, y)
            sum += loss.item()
            len += 1
        return sum/len


class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key   = nn.Linear(n_emb, head_size, bias=False)
        self.query = nn.Linear(n_emb, head_size, bias=False)
        self.value = nn.Linear(n_emb, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)   # (B, T, 16)
        q = self.query(x) # (B, T, 16)
        wei = q @ k.transpose(-2, -1) * C**-0.5 #--> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v

        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for i in range(n_heads)])
        self.proj = nn.Linear(n_emb, n_emb)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(self.dropout(out))
        return out

class FeedForward(nn.Module):
    def __init__(self, n_emb):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_emb, n_emb*4),
                                 nn.ReLU(),
                                 nn.Linear(n_emb*4, n_emb),
                                 nn.Dropout(dropout))
    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_emb, n_heads):
        super().__init__()
        head_size = n_emb//n_heads
        self.sa_heads = MultiHeadAttention(n_heads, head_size)
        self.ff = FeedForward(n_emb)
        self.ln1 = nn.LayerNorm(n_emb)
        self.ln2 = nn.LayerNorm(n_emb)

    def forward(self, x):
        x = x + self.sa_heads(self.ln1(x))
        x = x + self.ff(self.ln2(x))

        return x

class LanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_emb)
        self.position_embedding_table = nn.Embedding(block_size, n_emb)
        self.blocks = nn.Sequential(
                                    Block(n_emb, 4),
                                    Block(n_emb, 4),
                                    Block(n_emb, 4),
                                    Block(n_emb, 4),
                                    Block(n_emb, 4),
                                    Block(n_emb, 4),
                                    nn.LayerNorm(n_emb),
                                    )
        self.lm_head = nn.Linear(n_emb, vocab_size)

    def forward(self, x, y=None):
        B, T = x.shape
        token_emb = self.token_embedding_table(x)
        pos_embedding = self.position_embedding_table(torch.arange(T, device=device))
        x = token_emb + pos_embedding
        x = self.blocks(x)
        logits = self.lm_head(x)

        if y is not None:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            y = y.view(B * T)
            loss = F.cross_entropy(logits, y)
            return loss
        else:
            return logits

    def generate(self, idx, n_tokens):
        self.eval()
        for i in range(n_tokens):
            current_idx = idx[:, -block_size:]
            logits = self(current_idx)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=1)
            new_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, new_token), 1)
        return idx

    def _train(self, data, n_iter, lr):
        self.train()
        optim = torch.optim.AdamW(self.parameters(), lr)
        for i in range(n_iter):
            x, y = get_batch(data)
            loss = self(x, y)
            if i% 100 == 0:
                print(loss.item())

            optim.zero_grad()
            loss.backward()

            optim.step()
            if i% (n_iter/100) == 0:
                print(int(((i+1)/n_iter)*100))
        self.eval()

if __name__ == '__main__':
    with open('tokens.pkl', 'rb') as f:
        tokens = pickle.load(f)
    n = int(len(tokens) * 0.9)
    val_tokens = tokens[n:]
    val_tokens = torch.tensor(tokens).to(device)
    model = LanguageModel(vocab_size)
    model = model.to(device)
    model.load_state_dict(torch.load('parameters.pth', map_location=torch.device(device)))
    val_loss = get_loss(val_tokens, model)
    print(f'Validation loss is {val_loss}')