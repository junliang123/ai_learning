import torch 
import torch.nn as nn
from torch.nn import functional as F

batch_size = 32
block_size = 128
max_iters = 2000
eval_interval = 200
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 20
n_embd = 180
n_head = 6
n_layer = 4
dropout = 0.2

torch.manual_seed(1337)

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
dictionary_size = len(chars)
map_char_to_id = {ch:i for i, ch in enumerate(chars)}
map_id_to_char = {i:ch for i, ch in enumerate(chars)}

def map_string_to_list(s):
    return [map_char_to_id[c] for c in s]

def map_list_to_string(l):
    return ''.join([map_id_to_char[i] for i in l])

#print(text[:20])
#print(map_list_to_string(map_string_to_list(text[:20])))

training_set = torch.tensor(map_string_to_list(text[:(int)(0.9*len(text))]), dtype=torch.long)
validation_set = torch.tensor(map_string_to_list(text[(int)(0.9*len(text)):]), dtype=torch.long)

def data_sampling(set_type):
    data = training_set if set_type == 'train' else validation_set
    idx = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in idx])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in idx])
    x, y = x.to(device), y.to(device)
    return x, y

class SelfAttention(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.WQ = nn.Linear(n_embd, head_size, bias=False)
        self.WK = nn.Linear(n_embd, head_size, bias=False)
        self.WV = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        Q = self.WQ(x)
        K = self.WK(x)
        V = self.WV(x)
        O = Q @ K.transpose(-2, -1)
        O = O*(K.shape[-1] ** -0.5)
        O = O.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        O = F.softmax(O, dim=-1)
        O = O @ V
        return O

class MultiHeadAttention(nn.Module):
    def __init__(self, head_nums, head_size):
        super().__init__()
        self.heads = nn.ModuleList([SelfAttention(head_size) for _ in range(head_nums)])
        self.trans = nn.Linear(head_nums*head_size, n_embd)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        out = self.dropout(self.trans(out))
        return out
    
class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.layer1 = nn.Linear(n_embd, 4*n_embd)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(4*n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        out = self.layer1(x)
        out = self.relu(out)
        out = self.layer2(out)
        out = self.dropout(out)
        return out

class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        head_size = n_embd // n_head
        self.ln1 = nn.LayerNorm(n_embd)
        self.mutihead = MultiHeadAttention(n_head, head_size)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ff = FeedForward(n_embd)

    def forward(self, x):
        x = x + self.mutihead(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x
        

class GPTLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(dictionary_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)
        self.blocks = nn.ModuleList([TransformerBlock() for _ in range(n_layer)])
        self.ln = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, dictionary_size)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        logits = self.token_embedding(idx)
        pos = torch.arange(T, device=idx.device)
        logits = logits + self.position_embedding(pos)
        for block in self.blocks:
            logits = block(logits)
        logits = self.ln(logits)
        logits = self.lm_head(logits)
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            B, T = idx.shape
            logits, loss = self(idx[:, max(0, T - block_size):])
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
    
    @torch.no_grad()
    def estimate_loss(self, iter):
        self.eval()
        training_loss = 0.0
        validation_loss = 0.0
        for _ in range(eval_iters):
            xb, yb = data_sampling('train')
            logits, loss = self(xb, yb)
            training_loss += loss
            xb, yb = data_sampling('valid')
            logits, loss = self(xb, yb)
            validation_loss += loss
        training_loss /= eval_iters
        validation_loss /= eval_iters
        print(f"step {iter}:\n training_loss: {training_loss}\n validation_loss: {validation_loss}")
        self.train()

    def train_model(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=learning_rate/2.0)
        for iter in range(max_iters):
            xb, yb = data_sampling('train')
            logits, loss = self(xb, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            if iter % eval_interval == 0 or iter == max_iters - 1:
                self.estimate_loss(iter)

model = GPTLanguageModel().to(device)
model.train_model()

context = torch.zeros((1, 1), dtype=torch.long, device=device)
context = model.generate(context, 50)[0].tolist()
print(map_list_to_string(context))