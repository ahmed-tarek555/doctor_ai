import json
import tiktoken


tokenizer = tiktoken.get_encoding('gpt2')

vocab_size = 1000
def get_stats(tokens):
    counts = {}
    for pair in zip(tokens, tokens[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts

def merge(ids, pair, idx):
    newids = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
            newids.append(idx)
            i += 2
        else:
            newids.append(ids[i])
            i += 1
    return newids


class Tokenizer:
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size
        self.merges = {}
        self.vocab = {idx: bytes([idx]) for idx in range(256)}


    def __call__(self, text):
        tokens = text.encode('utf-8')
        tokens = list(map(int, tokens))
        num_merges = self.vocab_size - 256
        for i in range(num_merges):
            idx = 256+i
            stats = get_stats(tokens)
            top_pair = max(stats, key=stats.get)
            tokens = merge(tokens, top_pair, idx)
            self.merges[top_pair] = idx
        for (p0, p1), idx in self.merges.items():
            self.vocab[idx] = self.vocab[p0] + self.vocab[p1]
        return self.merges, self.vocab

    def decode(self, ids, vocab):
        unicode_bytes = b''.join(vocab[idx] for idx in ids)
        text = unicode_bytes.decode('utf-8', errors='replace')
        return text

    def encode(self, text, merges):
        tokens = text.encode('utf-8')
        tokens = list(map(int, tokens))
        while True:
            stats = get_stats(tokens)
            pair = min(stats, key=lambda p: merges.get(p, float('inf')))
            if pair not in merges:
                break
            idx = merges[pair]
            tokens = merge(tokens, pair, idx)
        return tokens


if __name__ == '__main__':
  print(tokenizer.encode('<bos>'))
