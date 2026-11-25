import os
import json
import yaml
import random
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

from tokenizer import KoreanTokenizer
from model import BiLSTMClassifier

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class NSMCDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = torch.tensor(sequences, dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]

def encode_texts(texts, tokenizer, vocab, max_len):
    unk_id = vocab["<UNK>"]
    pad_id = vocab["<PAD>"]
    encoded = []
    for doc in tqdm(texts, desc="Encoding"):
        tokens = tokenizer.tokenize(doc)
        ids = [vocab.get(t, unk_id) for t in tokens][:max_len]
        if len(ids) < max_len:
            ids += [pad_id] * (max_len - len(ids))
        encoded.append(ids)
    return encoded

def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y.numpy())
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro")
    return acc, f1

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, required=True, choices=["B", "C"])
    args = parser.parse_args()

    with open("config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    with open(cfg["vocab"]["vocab_path"], "r", encoding="utf-8") as f:
        vocab = json.load(f)

    proc_dir = cfg["data"]["processed_dir"]
    train_df = pd.read_csv(os.path.join(proc_dir, "train.csv"))
    valid_df = pd.read_csv(os.path.join(proc_dir, "valid.csv"))

    # Model B: 정규화 미적용(False) / Model C: 구어체 반복 문자 정규화 적용(True) - One Factor 비교
    use_norm = (args.model_type == "C")
    print(f"\n=== [Model {args.model_type}] 학습 시작 (normalize={use_norm}) ===")
    tokenizer = KoreanTokenizer(normalize=use_norm)

    max_len = cfg["model"]["max_len"]
    X_train = encode_texts(train_df["document"], tokenizer, vocab, max_len)
    X_valid = encode_texts(valid_df["document"], tokenizer, vocab, max_len)

    train_ds = NSMCDataset(X_train, train_df["label"].values)
    valid_ds = NSMCDataset(X_valid, valid_df["label"].values)

    batch_size = cfg["model"]["batch_size"]
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False)

    model = BiLSTMClassifier(
        vocab_size=len(vocab),
        embed_dim=cfg["model"]["embed_dim"],
        hidden_dim=cfg["model"]["hidden_dim"],
        num_layers=cfg["model"]["num_layers"],
        dropout=cfg["model"]["dropout"]
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["model"]["lr"])

    best_val_f1 = 0.0
    save_path = f"outputs/model_{args.model_type.lower()}_bilstm.pt"

    for epoch in range(1, cfg["model"]["epochs"] + 1):
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        val_acc, val_f1 = evaluate(model, valid_loader, device)
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch}/{cfg['model']['epochs']} | Loss: {avg_loss:.4f} | Val Acc: {val_acc*100:.2f}% | Val Macro-F1: {val_f1*100:.2f}%")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), save_path)

    print(f"=== [Model {args.model_type}] 완료 | Best Val Macro-F1: {best_val_f1*100:.2f}% -> {save_path} ===")

if __name__ == "__main__":
    main()
