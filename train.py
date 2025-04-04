import time
import math

import torch
from torch import nn
import torch.nn.functional as F
from datasets import load_dataset


device = "cuda" if torch.cuda.is_available() else "cpu"
dataset = load_dataset("muzaffercky/kurdish-kurmanji-articles", split="train")


class Lang:
    def __init__(self, text: str, is_word_level=False):
        self.is_word_level = is_word_level

        if is_word_level:
            words = set(
                word for sentence in text.split(".") for word in sentence.split(" ")
            )
            self.vocab_size = len(words)
            self.stoi = {word: i for i, word in enumerate(words)}
            self.itos = {i: word for i, word in enumerate(words)}
        else:
            chars = set(text)
            self.vocab_size = len(chars)
            self.stoi = {char: i for i, char in enumerate(chars)}
            self.itos = {i: char for i, char in enumerate(chars)}

    def encode(self, text: str):
        if not self.is_word_level:
            return [self.stoi[char] for char in text]

        words = [word for sentence in text.split(".") for word in sentence.split(" ")]
        return [self.stoi[word] for word in words]

    def decode(self, indexes):
        words = [self.itos[index.item()] for index in indexes]
        return " ".join(words)


def prepare_data(dataset):
    return "".join(row["content"] for row in dataset)


def get_batches(split):
    data = train_data if split == "train" else val_data
    indexes = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in indexes])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in indexes])
    x, y = x.to(device), y.to(device)
    return x, y


def to_minutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return f"{m:.4f} minutes {s:.4f} seconds"


def time_since(since, percent):
    now = time.time()
    s = now - since
    es = s / percent
    rs = es - s
    return f"spent: {to_minutes(s)} | remain: {to_minutes(rs)}"


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batches(split)
            _, loss = model(x, y)
            losses[k] = loss.item()

        out[split] = losses.mean()

    model.train()
    return out


class Head(nn.Module):

    def __init__(self, embd_size, head_size):
        super().__init__()
        self.W_key = nn.Linear(embd_size, head_size, bias=False)
        self.W_query = nn.Linear(embd_size, head_size, bias=False)
        self.W_value = nn.Linear(embd_size, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x):
        _, T, C = x.shape  # x shape is (B, T, C)

        key = self.W_key(x)  # (B, T, C) -> (B, T, head_size)
        query = self.W_query(x)  # (B, T, C) -> (B, T, head_size)

        # compute attention scores
        wei = (
            query @ key.transpose(-2, -1) * C**-0.5
        )  # (B, T, head_size) @ (B, head_size, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))  # (B, T, T)
        wei = F.softmax(wei, dim=-1)  # (B, T, T)
        wei = self.dropout(wei)  # (B, T, T)

        value = self.W_value(x)  # (B, T, C) -> (B, T, head_size)
        out = wei @ value  # (B, T, T) @ (B, T, head_size) -> (B, T, head_size)
        return out


class MultiHeadAttention(nn.Module):

    def __init__(self, embd_size, heads_size, heads_num):
        super().__init__()
        self.heads = nn.ModuleList(
            [Head(embd_size, heads_size) for _ in range(heads_num)]
        )
        self.proj = nn.Linear(embd_size, embd_size)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x):
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedForward(nn.Module):
    def __init__(self, embd_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embd_size, 4 * embd_size),
            nn.ReLU(),
            nn.Linear(4 * embd_size, embd_size),
            nn.Dropout(dropout_p),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):

    def __init__(self, embd_size, heads_num):
        super().__init__()
        head_size = embd_size // heads_num
        self.sa = MultiHeadAttention(embd_size, head_size, heads_num)
        self.ffwd = FeedForward(embd_size)
        self.ln1 = nn.LayerNorm(embd_size)
        self.ln2 = nn.LayerNorm(embd_size)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class GPTLanguage(nn.Module):

    def __init__(self, vocab_size, embd_size, heads_num, layers_num):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, embd_size)
        self.pos_embedding_table = nn.Embedding(block_size, embd_size)
        self.blocks = nn.Sequential(
            *[Block(embd_size, heads_num) for _ in range(layers_num)]
        )
        self.ln_f = nn.LayerNorm(embd_size)
        self.lm_head = nn.Linear(embd_size, vocab_size)

    def forward(self, idx, targets=None):
        T = idx.size(-1)
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.pos_embedding_table(torch.arange(T, device=idx.device))

        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            return logits, None

        B, T, C = logits.shape
        logits = logits.view(B * T, C)
        targets = targets.view(B * T)
        loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        # shape of idx is (B, T)

        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]  # Get tokens of last block_size
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]  # B, T, C -> B, C
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx


def train(model, optimizer, epochs_num, print_every):
    start = time.time()
    for epoch_num in range(1, epochs_num + 1):

        x, y = get_batches("train")
        optimizer.zero_grad()

        logits, loss = model(x, y)

        loss.backward()
        optimizer.step()

        if epoch_num % print_every == 0:
            losses = estimate_loss()
            time_ = time_since(start, epoch_num / epochs_num)
            print(
                f"epoch: {epoch_num} | train loss: {losses['train']:.4f} | val loss: {losses['val']:.4f} | {time_}"
            )


learning_rate = 1e-3
epochs_num = 100
eval_iters = 200
block_size = 16
batch_size = 16
dropout_p = 0.1
print_every = 10
heads_num = 8
embedding_size = 256
layers_num = 2


text = prepare_data(dataset)
lang = Lang(text, is_word_level=True)
data = torch.tensor(lang.encode(text), dtype=torch.long)
vocab_size = lang.vocab_size
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]


model = GPTLanguage(
    vocab_size=vocab_size,
    embd_size=embedding_size,
    heads_num=heads_num,
    layers_num=layers_num,
)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
print("vocab size:", vocab_size)
train(
    model,
    optimizer,
    epochs_num,
    print_every,
)

sentence = "ziman hebûn e"
encoded = lang.encode(sentence)
print("encoded:", encoded)
context = torch.tensor(encoded, dtype=torch.long).unsqueeze(0)
tokens = model.generate(context, max_new_tokens=200)
tokens_without_input = tokens[0][len(encoded) :]
generated = lang.decode(tokens_without_input)
print(generated)
