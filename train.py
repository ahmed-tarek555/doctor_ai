import torch
from utils import device, DataLoader, batch_size, block_size
from model import LanguageModel
from tokenizer import tokenizer

# n convo tokens = 26789159
# n data tokens  = 3077306

total_batch_size = 32768
grad_accum_steps = total_batch_size//(batch_size*block_size)
lr = 3e-4
n_iter = 8180
current_file = 'tokenized_chat_templated_convo'
torch.set_float32_matmul_precision('high')
model = LanguageModel(tokenizer.n_vocab)
model = model.to(device)
# model = torch.compile(model)

data_loader = DataLoader(current_file, batch_size, block_size)
model.train()
optim = torch.optim.AdamW(model.parameters(), lr)
for i in range(n_iter):
    loss_accum = 0
    optim.zero_grad()
    for micro_step in range(grad_accum_steps):
        x, y = data_loader.next_batch()
        x = x.to(device)
        y = y.to(device)
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            loss = model(x, y)
        loss = loss / grad_accum_steps
        loss_accum += loss.detach()
        loss.backward()
    if i % 10 == 0:
        print(loss_accum.item())
    optim.step()
    print(int(((i + 1) / n_iter) * 100))
model.eval()

save_file = 'chatbot_parameters.pth'
torch.save(model.state_dict(), save_file)
print(f'Parameters saved to {save_file}')
