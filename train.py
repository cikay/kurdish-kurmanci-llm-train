import time
import math
import random
import csv
import os
import argparse
from datetime import datetime

import torch
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from model import GPTLanguage
import sentencepiece as spm


tokenizer = spm.SentencePieceProcessor()
tokenizer.load(f"./tokenizers/kurmanji_unigram_10000.model")

if torch.cuda.is_available():
    device = "cuda"
    print("CUDA is used")
elif torch.backends.mps.is_available():
    device = "mps"
    print("MPS is used")
else:
    device = "cpu"
    print("CPU is used")


dataset = load_dataset("muzaffercky/kurdish-kurmanji-news")

train_dataset = dataset["train"]
validation_dataset = dataset["test"]

PATH = "."


parser = argparse.ArgumentParser(description="Fine tune model")
parser.add_argument("--file-name", type=str, help="Fine tune model file name")

args = parser.parse_args()

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
    lines = []
    non_empty_rows = 0
    for row in dataset:
        content = row.get("content")
        if content:
            lines.append(content)
        else:
            non_empty_rows += 1
    print(f"Number of non-empty rows: {non_empty_rows}")
    return "\n".join(lines)


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


def train(
    model,
    scheduler,
    train_loader,
    val_loader,
    optimizer,
    epochs_num,
    print_every,
    start_epoch=1,
):
    start = time.time()
    model.train()
    for epoch_num in range(start_epoch, epochs_num + 1):
        total_loss = 0
        total_batch = len(train_loader)
        print(f"epoch: {epoch_num} | total batches: {total_batch}")
        for batch_idx, (x, y) in enumerate(train_loader, 1):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()

            logits, loss = model(x, y)

            loss.backward()
            optimizer.step()
            scheduler.step()  # Step after each batch for more visible decay

            total_loss += loss.item()

            if batch_idx % print_every == 0:
                batch_loss = total_loss / batch_idx
                time_ = time_since(start, batch_idx / len(train_loader))
                print(
                    f"epoch: {epoch_num} | batch: {batch_idx}/{total_batch} | epoch loss: {batch_loss:.4f} | {time_}"
                )
                print(f"Epoch {epoch_num} | LR: {optimizer.param_groups[0]['lr']:.8f}")

            if batch_idx % 1000 == 0:
                model.eval()
                generated = generate()
                # save to csv file
                with open(f"{PATH}/generated_text.csv", "a") as f:
                    csv_writer = csv.writer(f)
                    csv_writer.writerow([generated])

                model.train()

        valid_loss = estimate_loss(model, val_loader)
        mean_loss = total_loss / total_batch
        print(
            f"epoch: {epoch_num} | average loss: {mean_loss:.4f} | validation loss: {valid_loss:.4f}"
        )

        model_name = generate_model_name(
            layers_num,
            epoch_num,
            block_size,
            embedding_size,
            heads_num,
        )

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "epoch": epoch_num,
                "scheduler_state_dict": scheduler.state_dict(),
                "hyperparameters": {
                    "embedding_size": embedding_size,
                    "heads_num": heads_num,
                    "layers_num": layers_num,
                    "block_size": block_size,
                    "vocab_size": vocab_size,
                    "learning_rate": optimizer.param_groups[0]['lr'],
                    "epochs_num": epochs_num,
                    "batch_size": batch_size,
                    "dropout_p": dropout_p,
                    "print_every": print_every,
                },
            },
            f"{PATH}/model_epoch_{epoch_num}.pth",
        )


def get_last_saved_model():
    max_epoch_file = ""
    max_epoch = 0
    all_entries = os.listdir(PATH)
    for file in all_entries:
        full_path = os.path.join(PATH, file)
        if not os.path.isfile(full_path) or not file.startswith("model_epoch_"):
            continue

        epoch_num = int(file.split("_")[2].split(".")[0])
        if epoch_num > max_epoch:
            max_epoch = epoch_num
            max_epoch_file = full_path

    return max_epoch_file


def get_model(max_epoch_file, layers_num = None):
    print("max_epoch_file", max_epoch_file)
    checkpoint = torch.load(max_epoch_file)
    print(f"Loading model from {max_epoch_file}")
    hyperparameters = checkpoint["hyperparameters"]
    model = GPTLanguage(
        vocab_size=hyperparameters["vocab_size"],
        embd_size=hyperparameters["embedding_size"],
        heads_num=hyperparameters["heads_num"],
        layers_num=layers_num or hyperparameters["layers_num"],
        block_size=hyperparameters["block_size"],
    ).to(device)
    missing = model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    print("Missing keys:", missing.missing_keys)
    print("Unexpected keys:", missing.unexpected_keys)
    print("Model loaded successfully")

    return model, checkpoint["epoch"]

def generate_model_name(layers_num, epoch_num, block_size, embedding_size, heads_num):
    date = datetime.now().strftime("%Y-%m-%d")
    return f"model_{date}_{epoch_num}e_{layers_num}l_{block_size}b_{embedding_size}em_{heads_num}h.pth"

@torch.no_grad()
def generate():
    sentence = "ziman hebûn e"
    encoded = tokenizer.Encode(sentence)
    context = torch.tensor(encoded, dtype=torch.long).unsqueeze(0).to(device)
    tokens = model.generate(context, max_new_tokens=2000)
    tokens_without_input = tokens[0][len(encoded) :]
    tokens_list = tokens_without_input.tolist()
    individual_tokens = tokenizer.decode(tokens_list)
    return "".join(individual_tokens)


learning_rate = 0.0007
epochs_num = 10
remaining_epochs = epochs_num
start_epoch = 1
eval_iters = 200
block_size = 100
batch_size = 16
dropout_p = 0.2
print_every = 10
heads_num = 8
embedding_size = 256
layers_num = 6


text = prepare_data(dataset)
vocab_size = tokenizer.get_piece_size()


print(f"File name: {args.file_name}")

if args.file_name:
    checkpoint = torch.load(args.file_name)
    print(f"Loading model from {args.file_name}")
    hyperparameters = checkpoint["hyperparameters"]
    model, last_saved_epoch = get_model(args.file_name, layers_num)
    learning_rate = hyperparameters["learning_rate"]
    remaining_epochs = epochs_num - last_saved_epoch
    start_epoch = last_saved_epoch + 1
    print("Model is loaded")
else:
    model = GPTLanguage(
        vocab_size=vocab_size,
        embd_size=embedding_size,
        heads_num=heads_num,
        layers_num=layers_num,
        block_size=block_size,
        dropout_p=dropout_p,
    ).to(device)
    print("Model initialized successfully")


optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

tokens = tokenizer.encode(text)
print(f"Number of tokens: {len(tokens)}")
data = torch.tensor(tokens, dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]


num_train_samples_per_epoch = len(train_data) // block_size
train_dataset = TextDataset(train_data, block_size, num_train_samples_per_epoch)

num_val_samples_per_epoch = len(val_data) // block_size
val_dataset = TextDataset(val_data, block_size, num_val_samples_per_epoch)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


# Replace CosineAnnealingLR with CosineAnnealingWarmRestarts for more visible decay
# T_0 is the number of iterations for the first restart cycle
# T_mult is a factor that increases T_0 after each restart
scheduler = CosineAnnealingWarmRestarts(
    optimizer,
    T_0=len(train_loader) // 10,  # First restart after ~10% of an epoch
    T_mult=2,  # Double the period after each restart
    eta_min=1e-7,  # Minimum learning rate
)

train(
    model,
    scheduler,
    train_loader,
    val_loader,
    optimizer,
    epochs_num,
    print_every,
    start_epoch=start_epoch,
)
