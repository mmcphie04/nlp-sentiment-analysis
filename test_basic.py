# test_basic_imdb.py

import os
import json
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import matplotlib.pyplot as plt
from tqdm import tqdm

from model import SentimentTransformer

# ---- Configuration ----
SPLIT_DIR     = "imdb_splits_90_10"
MODEL_NAME    = "distilbert-base-uncased"
BATCH_SIZE    = 16
MAX_LENGTH    = 128
# -EMBED_DIM     = 128
# -FF_DIM        = 256
# -NUM_LAYERS    = 2
# -DROPOUT       = 0.1
EMBED_DIM     = 64
FF_DIM        = 128
NUM_LAYERS    = 2
DROPOUT       = 0.3
NUM_EPOCHS    = 5
LEARNING_RATE = 5e-5
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- Data ----
class IMDBJsonDataset(Dataset):
    def __init__(self, json_path: str, tokenizer, max_length: int):
        with open(json_path, "r") as f:
            self.examples = json.load(f)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        encoding = self.tokenizer(
            ex["review"],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            add_special_tokens=True,
        )
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)
        label = torch.tensor(ex["label"], dtype=torch.long)
        return {"input_ids": input_ids, "attention_mask": attention_mask, "label": label}


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---- Training / Evaluation loops ----
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch in tqdm(loader, desc="  Training", leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / total
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


def eval_one_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc="  Evaluating", leave=False):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)

            total_loss += loss.item() * labels.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / total
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


# ---- Main ----
def main():
    print(f"Using device: {DEVICE}\n")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Prepare datasets & loaders
    train_ds = IMDBJsonDataset(os.path.join(SPLIT_DIR, "train.json"), tokenizer, MAX_LENGTH)
    test_ds  = IMDBJsonDataset(os.path.join(SPLIT_DIR, "test.json"),  tokenizer, MAX_LENGTH)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # Instantiate model
    model = SentimentTransformer(
        vocab_size=tokenizer.vocab_size,
        embed_dim=EMBED_DIM,
        ff_dim=FF_DIM,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        max_length=MAX_LENGTH
    ).to(DEVICE)

    print("Model Summary:")
    print(model)
    print(f"\nTotal trainable parameters: {count_parameters(model):,}\n")

    # Optimizer & loss
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    # Storage for metrics
    train_losses, train_accs = [], []
    test_losses,  test_accs  = [], []

    # Training loop
    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"Epoch {epoch}/{NUM_EPOCHS}")

        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
        test_loss,  test_acc  = eval_one_epoch(model, test_loader,  criterion, DEVICE)

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_losses.append(test_loss)
        test_accs.append(test_acc)

        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Test  Loss: {test_loss:.4f}, Test  Acc: {test_acc:.2f}%\n")

    # ---- Print out the four metric lists ----
    print("train_losses:", train_losses)
    print("train_accs:   ", train_accs)
    print("test_losses:  ", test_losses)
    print("test_accs:    ", test_accs)

    # ---- Plotting ----
    epochs = list(range(1, NUM_EPOCHS + 1))

    plt.figure()
    plt.plot(epochs, train_losses, label="Train loss")
    plt.plot(epochs, test_losses,  label="Test loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss by epoch")
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(epochs, train_accs, label="Train accuracy")
    plt.plot(epochs, test_accs,  label="Test accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Acc by epoch")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
