import os
import json
import yaml
import pickle
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from scipy.sparse import lil_matrix
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

from tokenizer import KoreanTokenizer
from model import BiLSTMClassifier
from train_dl import NSMCDataset, encode_texts

def texts_to_bow(texts, tokenizer, vocab):
    mat = lil_matrix((len(texts), len(vocab)), dtype=np.float32)
    unk_id = vocab["<UNK>"]
    for row_idx, doc in enumerate(texts):
        for tok in tokenizer.tokenize(doc):
            tok_id = vocab.get(tok, unk_id)
            if tok_id != 0:
                mat[row_idx, tok_id] += 1.0
    return mat.tocsr()

def predict_dl(model, loader, device):
    model.eval()
    all_preds, all_probs = [], []
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            logits = model(x)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)
            all_preds.extend(preds)
            all_probs.extend(probs[:, 1])
    return np.array(all_preds), np.array(all_probs)

def main():
    with open("config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(cfg["vocab"]["vocab_path"], "r", encoding="utf-8") as f:
        vocab = json.load(f)

    proc_dir = cfg["data"]["processed_dir"]
    test_df = pd.read_csv(os.path.join(proc_dir, "test.csv"))
    y_test = test_df["label"].values

    tok_raw = KoreanTokenizer(normalize=False)
    tok_norm = KoreanTokenizer(normalize=True)

    # 1. Model A (Naive Bayes) Test 평가
    with open("outputs/model_a_nb.pkl", "rb") as f:
        model_a = pickle.load(f)
    X_test_bow = texts_to_bow(test_df["document"], tok_raw, vocab)
    preds_a = model_a.predict(X_test_bow)
    acc_a = accuracy_score(y_test, preds_a)
    f1_a = f1_score(y_test, preds_a, average="macro")

    # 2. Model B (Bi-LSTM, normalize=False) Test 평가
    max_len = cfg["model"]["max_len"]
    batch_size = cfg["model"]["batch_size"]

    X_test_b = encode_texts(test_df["document"], tok_raw, vocab, max_len)
    loader_b = DataLoader(NSMCDataset(X_test_b, y_test), batch_size=batch_size, shuffle=False)

    model_b = BiLSTMClassifier(
        vocab_size=len(vocab),
        embed_dim=cfg["model"]["embed_dim"],
        hidden_dim=cfg["model"]["hidden_dim"],
        num_layers=cfg["model"]["num_layers"],
        dropout=cfg["model"]["dropout"]
    ).to(device)
    model_b.load_state_dict(torch.load("outputs/model_b_bilstm.pt", map_location=device))
    preds_b, probs_b = predict_dl(model_b, loader_b, device)
    acc_b = accuracy_score(y_test, preds_b)
    f1_b = f1_score(y_test, preds_b, average="macro")

    # 3. Model C (Bi-LSTM, normalize=True) Test 평가
    X_test_c = encode_texts(test_df["document"], tok_norm, vocab, max_len)
    loader_c = DataLoader(NSMCDataset(X_test_c, y_test), batch_size=batch_size, shuffle=False)

    model_c = BiLSTMClassifier(
        vocab_size=len(vocab),
        embed_dim=cfg["model"]["embed_dim"],
        hidden_dim=cfg["model"]["hidden_dim"],
        num_layers=cfg["model"]["num_layers"],
        dropout=cfg["model"]["dropout"]
    ).to(device)
    model_c.load_state_dict(torch.load("outputs/model_c_bilstm.pt", map_location=device))
    preds_c, _ = predict_dl(model_c, loader_c, device)
    acc_c = accuracy_score(y_test, preds_c)
    f1_c = f1_score(y_test, preds_c, average="macro")

    # 실험 비교표 저장
    rows = [
        {
            "Model": "Model A (Naive Bayes Baseline)",
            "Factor": "BoW + MultinomialNB",
            "Test_Accuracy(%)": round(acc_a * 100, 2),
            "Test_Macro_F1(%)": round(f1_a * 100, 2)
        },
        {
            "Model": "Model B (Bi-LSTM)",
            "Factor": "Kiwi 형태소 (정규화 X)",
            "Test_Accuracy(%)": round(acc_b * 100, 2),
            "Test_Macro_F1(%)": round(f1_b * 100, 2)
        },
        {
            "Model": "Model C (Bi-LSTM + Norm)",
            "Factor": "Kiwi 형태소 + 구어체 반복문자 정규화",
            "Test_Accuracy(%)": round(acc_c * 100, 2),
            "Test_Macro_F1(%)": round(f1_c * 100, 2)
        }
    ]
    results_df = pd.DataFrame(rows)
    results_df.to_csv("outputs/experiment_results.csv", index=False, encoding="utf-8-sig")
    print("\n=== [최종 Test 비교 실험 결과] ===")
    print(results_df.to_string(index=False))

    # 혼동 행렬 출력 (Best Model B 기준)
    cm = confusion_matrix(y_test, preds_b)
    print("\n=== [Model B 혼동 행렬 (Confusion Matrix)] ===")
    print(f"실제 부정(0) -> 예측 부정: {cm[0][0]:,} | 예측 긍정(오답): {cm[0][1]:,}")
    print(f"실제 긍정(1) -> 예측 부정(오답): {cm[1][0]:,} | 예측 긍정: {cm[1][1]:,}")

    # 직접 구현 추가 기능: 리뷰 토큰 길이별 성능 분석
    token_lens = [len(tok_raw.tokenize(doc)) for doc in test_df["document"]]
    test_df["token_len"] = token_lens
    test_df["pred_b"] = preds_b
    test_df["prob_b"] = probs_b

    bins = [0, 5, 15, 30, 999]
    labels_bin = ["단문 (1~5토큰)", "중단문 (6~15토큰)", "중장문 (16~30토큰)", "장문 (31토큰 이상)"]
    test_df["len_group"] = pd.cut(test_df["token_len"], bins=bins, labels=labels_bin)

    len_rows = []
    for grp in labels_bin:
        sub = test_df[test_df["len_group"] == grp]
        if len(sub) > 0:
            g_acc = accuracy_score(sub["label"], sub["pred_b"]) * 100
            g_f1 = f1_score(sub["label"], sub["pred_b"], average="macro") * 100
            len_rows.append({
                "길이 구간": grp,
                "샘플 수": len(sub),
                "Accuracy(%)": round(g_acc, 2),
                "Macro-F1(%)": round(g_f1, 2)
            })

    len_df = pd.DataFrame(len_rows)
    len_df.to_csv("outputs/length_analysis.csv", index=False, encoding="utf-8-sig")
    print("\n=== [추가 평가: 문장 길이별 성능 분석 (Model B)] ===")
    print(len_df.to_string(index=False))

    # 오답 사례 10개 추출
    wrong_df = test_df[test_df["label"] != test_df["pred_b"]].copy()
    wrong_df["confidence"] = np.where(wrong_df["pred_b"] == 1, wrong_df["prob_b"], 1 - wrong_df["prob_b"])
    top_errors = wrong_df.sort_values(by="confidence", ascending=False).head(10)

    with open("outputs/error_analysis.md", "w", encoding="utf-8") as f:
        f.write("# 오류 사례 10개 분석 (Model B 기준)\n\n")
        for idx, (_, row) in enumerate(top_errors.iterrows(), start=1):
            tokens = tok_raw.tokenize(row["document"])
            unk_cnt = sum(1 for t in tokens if t not in vocab)
            true_str = "긍정(1)" if row["label"] == 1 else "부정(0)"
            pred_str = "긍정(1)" if row["pred_b"] == 1 else "부정(0)"
            f.write(f"### 사례 {idx}\n")
            f.write(f"- **원문**: {row['document']}\n")
            f.write(f"- **토큰화 결과**: {tokens} (UNK 개수: {unk_cnt}개)\n")
            f.write(f"- **정답**: {true_str} / **예측**: {pred_str} (예측 확신도: {row['confidence']*100:.1f}%)\n")
            f.write("- **추정 원인**: 반어법/비꼬기 표현, 신조어·오탈자로 인한 UNK 발생, 긍정·부정 어휘 혼재\n\n")

    print("\n[완료] 오답 사례 10개가 outputs/error_analysis.md 에 저장되었습니다.")

if __name__ == "__main__":
    main()
