import time
import math
import random
import csv

import torch
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader


from tokenization import Lang
from model import GPTLanguage


device = "cuda" if torch.cuda.is_available() else "cpu"
dataset = load_dataset("muzaffercky/kurdish-kurmanji-articles", split="train")


class TextDataset(Dataset):
    def __init__(self, data, block_size, num_samples_per_epoch):
        self.data = data
        self.block_size = block_size
        self.num_samples_per_epoch = num_samples_per_epoch
        self.max_start_index = len(self.data) - self.block_size

    def __len__(self):
        return self.num_samples_per_epoch

    def __getitem__(self, idx):
        start_idx = random.randint(0, self.max_start_index)
        end_idx = start_idx + self.block_size
        x = self.data[start_idx:end_idx]
        y = self.data[start_idx + 1 : end_idx + 1]
        return x, y


def prepare_data(dataset):
    text = ""
    non_empty_rows = 0
    for row in dataset:
        if row["content"] is not None:
            text += row["content"] + "\n"
        else:
            non_empty_rows += 1
    print(f"Number of non-empty rows: {non_empty_rows}")
    return text


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
def estimate_loss(model, val_loader):
    model.eval()
    valid_data_length = len(val_loader)
    losses = torch.zeros(valid_data_length)
    print(f"Validation data length: {valid_data_length}")
    for k, (x, y) in enumerate(val_loader):
        x, y = x.to(device), y.to(device)
        _, loss = model(x, y)
        losses[k] = loss.item()

    model.train()
    return losses.mean()


def train(model, train_loader, val_loader, optimizer, epochs_num, print_every):
    start = time.time()
    model.train()
    for epoch_num in range(1, epochs_num + 1):
        total_loss = 0
        total_batch = len(train_loader)
        print(f"epoch: {epoch_num} | total batches: {total_batch}")
        for batch_idx, (x, y) in enumerate(train_loader, 1):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()

            logits, loss = model(x, y)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            if batch_idx % print_every == 0:
                batch_loss = total_loss / batch_idx
                time_ = time_since(start, batch_idx / len(train_loader))
                print(
                    f"epoch: {epoch_num} | batch: {batch_idx}/{total_batch} | epoch loss: {batch_loss:.4f} | {time_}"
                )

            if batch_idx % 1000 == 0:
                model.eval()
                generated = generate()
                # save to csv file
                with open("generated_text.csv", "a") as f:
                    csv_writer = csv.writer(f)
                    csv_writer.writerow([generated])

                model.train()

        valid_loss = estimate_loss(model, val_loader)
        mean_loss = total_loss / total_batch
        print(
            f"epoch: {epoch_num} | average loss: {mean_loss:.4f} | validation loss: {valid_loss:.4f}"
        )

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "hyperparameters": {
                    "embedding_size": embedding_size,
                    "heads_num": heads_num,
                    "layers_num": layers_num,
                    "block_size": block_size,
                    "vocab_size": vocab_size,
                    "vocab": lang.vocab,
                    "learning_rate": learning_rate,
                    "epochs_num": epochs_num,
                    "batch_size": batch_size,
                    "dropout_p": dropout_p,
                    "print_every": print_every,
                },
            },
            f"model_epoch_{epoch_num}.pth",
        )


@torch.no_grad()
def generate():
    sentence = "ziman hebûn e"
    encoded = lang.encode(sentence)
    context = torch.tensor(encoded, dtype=torch.long).unsqueeze(0)
    tokens = model.generate(context, max_new_tokens=2000)
    tokens_without_input = tokens[0][len(encoded) :]
    return lang.decode(tokens_without_input)


learning_rate = 0.0007
epochs_num = 100
eval_iters = 200
block_size = 320
batch_size = 16
dropout_p = 0.1
print_every = 10
heads_num = 16
embedding_size = 256
layers_num = 4


text = prepare_data(dataset)
lang = Lang(set(text))
data = torch.tensor(lang.encode(text), dtype=torch.long)
vocab_size = lang.vocab_size
print(f"vocab: {''.join(lang.vocab)}")
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]


num_train_samples_per_epoch = len(train_data) // block_size
train_dataset = TextDataset(train_data, block_size, num_train_samples_per_epoch)

num_val_samples_per_epoch = len(val_data) // block_size
val_dataset = TextDataset(val_data, block_size, num_val_samples_per_epoch)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


model = GPTLanguage(
    vocab_size=vocab_size,
    embd_size=embedding_size,
    heads_num=heads_num,
    layers_num=layers_num,
    block_size=block_size,
    dropout_p=dropout_p,
)


optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
print("vocab size:", vocab_size)

train(
    model,
    train_loader,
    val_loader,
    optimizer,
    epochs_num,
    print_every,
)
