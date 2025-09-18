import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
from tokenizer import tokenizer
from utils import get_loss, block_size, device

n_emb = 512
n_heads = 8
dropout = 0.2

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
        k = self.key(x)   # (B, T, head_size)
        q = self.query(x) # (B, T, head_size)
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
                                    Block(n_emb, 4),
                                    Block(n_emb, 4),
                                    nn.LayerNorm(n_emb),
                                    )
        self.lm_head = nn.Linear(n_emb, vocab_size)

        self.token_embedding_table.weight = self.lm_head.weight

        self.apply(self._init_weights)
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0, std=0.02)

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

if __name__ == '__main__':
    with open('tokenized_data.pkl', 'rb') as f:
        tokens = pickle.load(f)
    n = int(len(tokens) * 0.9)
    val_tokens = tokens[n:]
    val_tokens = torch.tensor(tokens).to(device)
    model = LanguageModel(tokenizer.n_vocab)
    total_params = sum(p.numel() for p in model.parameters())
    print(f'The number of parameters is: {total_params}')
    #The number of parameters is: 51120721
    model = model.to(device)
    # model = torch.compile(model)
    model.load_state_dict(torch.load('parameters.pth'))
    val_loss = get_loss(val_tokens, model)
    print(f'Validation loss is {val_loss}')