import torch
from utils import device, block_size
from model import LanguageModel
from tokenizer import tokenizer

model = LanguageModel(tokenizer.n_vocab)
model = model.to(device)
# model = torch.compile(model)
model.load_state_dict(torch.load('chatbot_parameters.pth'))

def format_example(example):
    return f"{example['role']}: {example['content']}\n"

context = [
    {'role': 'context_filler', 'content': ''.join(' ' for i in range(block_size))}
]

while True:

    prompt = input()
    if prompt.lower() == 'bye':
        break
    context.append({'role': 'Human', 'content': f'{prompt}'})

    chat = ''.join([format_example(message) for message in context])
    chat += 'Ai:'

    input_tokens = torch.tensor(tokenizer.encode(chat), dtype=torch.long).unsqueeze(0).to(device)

    generated_tokens = model.generate(input_tokens, 50)[0].tolist()
    generated_tokens = generated_tokens[input_tokens.shape[-1]:]
    generated_text = tokenizer.decode(generated_tokens)
    context.append({'role': 'Ai', 'content': f'{generated_text}'})

    print(f'{generated_text}\n')