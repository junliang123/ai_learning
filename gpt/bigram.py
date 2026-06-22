from pathlib import Path

import torch
import torch.nn as nn
from torch.nn import functional as F


torch.manual_seed(1337)

BATCH_SIZE = 32
BLOCK_SIZE = 8
MAX_STEPS = 10000
LEARNING_RATE = 1e-3
GENERATE_TOKENS = 100


class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        logits = self.token_embedding_table(idx)

        loss = None
        if targets is not None:
            b, t, c = logits.shape
            logits_for_loss = logits.view(b * t, c)
            targets = targets.view(b * t)
            loss = F.cross_entropy(logits_for_loss, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, _ = self(idx)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


def build_vocab(text):
    chars = sorted(list(set(text)))
    char_to_id = {ch: i for i, ch in enumerate(chars)}
    id_to_char = {i: ch for i, ch in enumerate(chars)}
    return chars, char_to_id, id_to_char


def encode(text, char_to_id):
    return [char_to_id[ch] for ch in text]


def decode(ids, id_to_char):
    return "".join(id_to_char[i] for i in ids)


def get_batch(split, train_data, val_data):
    data = train_data if split == "train" else val_data
    positions = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,))
    x = torch.stack([data[i : i + BLOCK_SIZE] for i in positions])
    y = torch.stack([data[i + 1 : i + BLOCK_SIZE + 1] for i in positions])
    return x, y


def main():
    input_path = Path(__file__).with_name("input.txt")
    text = input_path.read_text(encoding="utf-8")

    chars, char_to_id, id_to_char = build_vocab(text)
    vocab_size = len(chars)
    data = torch.tensor(encode(text, char_to_id), dtype=torch.long)

    split_index = int(0.9 * len(data))
    train_data = data[:split_index]
    val_data = data[split_index:]

    model = BigramLanguageModel(vocab_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    for step in range(MAX_STEPS):
        xb, yb = get_batch("train", train_data, val_data)
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step % 1000 == 0:
            print(f"step {step}: loss {loss.item()}")

    idx = torch.zeros((1, 1), dtype=torch.long)
    generated_ids = model.generate(idx, max_new_tokens=GENERATE_TOKENS)[0].tolist()
    print(decode(generated_ids, id_to_char))


if __name__ == "__main__":
    main()
