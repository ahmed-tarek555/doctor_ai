import torch
import json
import pickle

batch_size = 16
block_size = 256
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class DataLoader:
    def __init__(self, tokens_file, batch_size, block_size):
        self.B = batch_size
        self.T = block_size
        with open(tokens_file, 'rb') as f:
            tokens = json.load(f)
        print(f'{len(tokens)} tokens loaded')
        self.tokens = torch.tensor(tokens)
        self.current_position = 0

    def next_batch(self):
        buffer = self.tokens[self.current_position : self.current_position +self.B*self.T+1]
        x = buffer[:-1].view(self.B, self.T)
        y = buffer[1:].view(self.B, self.T)
        self.current_position += self.B*self.T
        if self.current_position + (self.B*self.T+1) > len(self.tokens):
            self.current_position = 0
        return x, y

with torch.no_grad():
    def get_loss(current_file, batch_size, block_size, model):
        data_loader = DataLoader(current_file, batch_size, block_size)
        model.eval()
        sum = 0
        len = 0
        for i in range((batch_size*block_size+1) * 3):
            x, y = data_loader.next_batch()
            loss = model(x, y)
            sum += loss.item()
            len += 1
        return sum/len