import os
import json
import yaml
import pandas as pd
from collections import Counter
from tqdm import tqdm
from tokenizer import KoreanTokenizer

def main():
    with open("config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    proc_dir = cfg["data"]["processed_dir"]
    min_freq = cfg["vocab"]["min_freq"]
    max_vocab_size = cfg["vocab"]["max_vocab_size"]
    vocab_path = cfg["vocab"]["vocab_path"]

    os.makedirs(os.path.dirname(vocab_path), exist_ok=True)

    train_df = pd.read_csv(os.path.join(proc_dir, "train.csv"))
    valid_df = pd.read_csv(os.path.join(proc_dir, "valid.csv"))

    tokenizer = KoreanTokenizer(normalize=False)

    # 1. 오직 train 데이터에서만 토큰 빈도 계산
    print("[1/3] Train 데이터 토큰화 및 빈도 계산 중...")
    counter = Counter()
    for doc in tqdm(train_df["document"], desc="Train Tokenizing"):
        tokens = tokenizer.tokenize(doc)
        counter.update(tokens)

    # 2. min_freq 이상 및 max_vocab_size 이내로 어휘 사전 구축
    vocab = {"<PAD>": 0, "<UNK>": 1}
    sorted_tokens = [
        tok for tok, freq in counter.most_common()
        if freq >= min_freq
    ][: max_vocab_size - len(vocab)]

    for idx, tok in enumerate(sorted_tokens, start=len(vocab)):
        vocab[tok] = idx

    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)

    print(f"\n[2/3] 어휘 구축 완료 -> {vocab_path}")
    print(f"  - 토큰화 방법: Kiwi 형태소 분석기 (표면형 단위)")
    print(f"  - 전체 Train 고유 토큰 수: {len(counter):,}개")
    print(f"  - 최소 빈도(min_freq): {min_freq}")
    print(f"  - 최종 어휘 크기(특수 토큰 포함): {len(vocab):,}개")

    # 3. 문장 -> 토큰 -> ID 변환 예시 3개 출력 (보고서 작성용)
    print("\n=== 문장 -> 토큰 -> ID 변환 예시 3개 ===")
    unk_id = vocab["<UNK>"]
    for i in range(3):
        sample_text = train_df["document"].iloc[i]
        tokens = tokenizer.tokenize(sample_text)
        token_ids = [vocab.get(t, unk_id) for t in tokens]
        print(f"[예시 {i+1}]")
        print(f"  원문 : {sample_text}")
        print(f"  토큰 : {tokens}")
        print(f"  ID   : {token_ids}")

    # 4. Validation 데이터의 미등록어(<UNK>) 비율 계산
    print("\n[3/3] Validation 미등록어(<UNK>) 비율 계산 중...")
    total_val_tokens = 0
    unk_val_tokens = 0
    for doc in tqdm(valid_df["document"], desc="Valid UNK Check"):
        tokens = tokenizer.tokenize(doc)
        total_val_tokens += len(tokens)
        unk_val_tokens += sum(1 for t in tokens if t not in vocab)

    unk_ratio = (unk_val_tokens / total_val_tokens) * 100 if total_val_tokens > 0 else 0.0
    print(f"=== Validation 미등록어(<UNK>) 통계 ===")
    print(f"  - 전체 검증 토큰 수: {total_val_tokens:,}개")
    print(f"  - 미등록어(<UNK>) 수: {unk_val_tokens:,}개")
    print(f"  - 미등록어(<UNK>) 비율: {unk_ratio:.2f}%")

if __name__ == "__main__":
    main()
