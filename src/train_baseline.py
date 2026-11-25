import os
import json
import yaml
import pickle
import numpy as np
import pandas as pd
from scipy.sparse import lil_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
from tokenizer import KoreanTokenizer

def texts_to_bow(texts, tokenizer, vocab):
    num_docs = len(texts)
    vocab_size = len(vocab)
    mat = lil_matrix((num_docs, vocab_size), dtype=np.float32)
    unk_id = vocab["<UNK>"]

    for row_idx, doc in enumerate(tqdm(texts, desc="BoW Vectorizing")):
        tokens = tokenizer.tokenize(doc)
        for tok in tokens:
            tok_id = vocab.get(tok, unk_id)
            if tok_id != 0:  # <PAD> 제외
                mat[row_idx, tok_id] += 1.0
    return mat.tocsr()

def main():
    with open("config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    proc_dir = cfg["data"]["processed_dir"]
    vocab_path = cfg["vocab"]["vocab_path"]

    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)

    train_df = pd.read_csv(os.path.join(proc_dir, "train.csv"))
    valid_df = pd.read_csv(os.path.join(proc_dir, "valid.csv"))

    tokenizer = KoreanTokenizer(normalize=False)

    X_train = texts_to_bow(train_df["document"], tokenizer, vocab)
    y_train = train_df["label"].values

    X_valid = texts_to_bow(valid_df["document"], tokenizer, vocab)
    y_valid = valid_df["label"].values

    # Baseline A: Multinomial Naive Bayes
    model_a = MultinomialNB(alpha=1.0)
    model_a.fit(X_train, y_train)

    val_preds = model_a.predict(X_valid)
    val_acc = accuracy_score(y_valid, val_preds)
    val_f1 = f1_score(y_valid, val_preds, average="macro")

    os.makedirs("outputs", exist_ok=True)
    with open("outputs/model_a_nb.pkl", "wb") as f:
        pickle.dump(model_a, f)

    print("\n=== [Model A: Naive Bayes Baseline] Validation 결과 ===")
    print(f"  - Validation Accuracy : {val_acc * 100:.2f}%")
    print(f"  - Validation Macro-F1 : {val_f1 * 100:.2f}%")
    print("  - 모델 저장 완료: outputs/model_a_nb.pkl")

if __name__ == "__main__":
    main()
  
