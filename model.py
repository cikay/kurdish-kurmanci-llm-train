import torch
from torch import nn
import torch.nn.functional as F



class Head(nn.Module):

    def __init__(self, embd_size, head_size, block_size, dropout_p=0.1):
        super().__init__()
        self.W_key = nn.Linear(embd_size, head_size, bias=False)
        self.W_query = nn.Linear(embd_size, head_size, bias=False)
        self.W_value = nn.Linear(embd_size, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x):
        _, T, C = x.shape  # x has shape (batch size, block size, embedding_size)
        key = self.W_key(x)
        query = self.W_query(x)  # (B, T, C) -> (B, T, head_size)

        # compute attention scores
        wei = (
            query @ key.transpose(-2, -1) * C**-0.5
        )  # (B, T, C) @ (B, C, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))  # (B, T, T)
        wei = F.softmax(wei, dim=-1)  # (B, T, T)
        wei = self.dropout(wei)  # (B, T, T)

        value = self.W_value(x)  # (B, T, C) -> (B, T, head_size)
        out = wei @ value  # (B, T, T) @ (B, T, head_size) -> (B, T, head_size)
        return out


class MultiHeadAttention(nn.Module):

    def __init__(self, embd_size, heads_size, heads_num, block_size, dropout_p=0.1):
        super().__init__()
        self.heads = nn.ModuleList(
            [
                Head(embd_size, heads_size, block_size, dropout_p)
                for _ in range(heads_num)
            ]
        )
        self.proj = nn.Linear(embd_size, embd_size)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x):
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedForward(nn.Module):
    def __init__(self, embd_size, dropout_p=0.1):
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

    def __init__(self, embd_size, heads_num, block_size, dropout_p=0.1):
        super().__init__()
        head_size = embd_size // heads_num
        self.sa = MultiHeadAttention(
            embd_size, head_size, heads_num, block_size, dropout_p
        )
        self.ffwd = FeedForward(embd_size, dropout_p)
        self.ln1 = nn.LayerNorm(embd_size)
        self.ln2 = nn.LayerNorm(embd_size)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class GPTLanguage(nn.Module):

    def __init__(
        self, vocab_size, embd_size, heads_num, layers_num, block_size, dropout_p=0.1
    ):
        super().__init__()
        self.block_size = block_size
        self.token_embedding_table = nn.Embedding(vocab_size, embd_size)
        self.pos_embedding_table = nn.Embedding(block_size, embd_size)
        self.blocks = nn.Sequential(
            *[
                Block(embd_size, heads_num, block_size, dropout_p)
                for _ in range(layers_num)
            ]
        )
        self.ln_f = nn.LayerNorm(embd_size)
        self.lm_head = nn.Linear(embd_size, vocab_size)

    def forward(self, idx, targets=None):
        T = idx.size(-1)
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.pos_embedding_table(
            torch.arange(T, device=idx.device)
        )  # (T) -> (T, embd_size)

        x = tok_emb + pos_emb  # (B, T, embd_size) + (T, embd_size) -> (B, T, embd_size)
        x = self.blocks(x)  # (B, T, embd_size) -> (B, T, embd_size)
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
            idx_cond = idx[:, -self.block_size :]  # Get tokens of last block_size
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]  # B, T, C -> B, C
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
