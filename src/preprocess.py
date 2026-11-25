import os
import yaml
import requests
import pandas as pd
from sklearn.model_selection import train_test_split

def download_file(url: str, save_path: str):
    if os.path.exists(save_path):
        print(f"[Skip] 이미 파일이 존재합니다: {save_path}")
        return
    print(f"[Download] 다운로드 중: {url}")
    res = requests.get(url)
    res.raise_for_status()
    with open(save_path, "wb") as f:
        f.write(res.content)
    print(f"[Done] 저장 완료: {save_path}")

def clean_dataframe(df: pd.DataFrame, name: str) -> pd.DataFrame:
    initial_len = len(df)
    # 1. 결측치 및 공백만 있는 빈 문장 제거
    df = df.dropna(subset=["document", "label"]).copy()
    df["document"] = df["document"].astype(str).str.strip()
    df = df[df["document"] != ""]
    after_empty_len = len(df)

    # 2. 완전 중복 리뷰 제거 (동일 원문이 다른 분할에 섞이는 데이터 누수 방지)
    df = df.drop_duplicates(subset=["document"]).reset_index(drop=True)
    after_dup_len = len(df)

    print(f"--- [{name}] 전처리 요약 ---")
    print(f"  원본 개수: {initial_len:,}")
    print(f"  빈 문장 제거 후: {after_empty_len:,} (-{initial_len - after_empty_len:,})")
    print(f"  중복 제거 후: {after_dup_len:,} (-{after_empty_len - after_dup_len:,})")
    return df[["document", "label"]]

def main():
    with open("config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    seed = cfg["seed"]
    raw_dir = cfg["data"]["raw_dir"]
    proc_dir = cfg["data"]["processed_dir"]
    valid_ratio = cfg["data"]["valid_ratio"]

    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(proc_dir, exist_ok=True)

    train_url = "https://raw.githubusercontent.com/e9t/nsmc/master/ratings_train.txt"
    test_url = "https://raw.githubusercontent.com/e9t/nsmc/master/ratings_test.txt"
    train_raw_path = os.path.join(raw_dir, "ratings_train.txt")
    test_raw_path = os.path.join(raw_dir, "ratings_test.txt")

    download_file(train_url, train_raw_path)
    download_file(test_url, test_raw_path)

    train_df = pd.read_csv(train_raw_path, sep="\t")
    test_df = pd.read_csv(test_raw_path, sep="\t")

    train_df = clean_dataframe(train_df, "Train 원본")
    test_df = clean_dataframe(test_df, "Test 원본")

    # Test 셋에 Train과 겹치는 문장이 있다면 Test에서 제거 (데이터 누수 원천 차단)
    train_docs = set(train_df["document"])
    before_overlap = len(test_df)
    test_df = test_df[~test_df["document"].isin(train_docs)].reset_index(drop=True)
    print(f"--- [Test] Train과 중복된 문장 제거: -{before_overlap - len(test_df):,} ---")

    # Train -> Train / Valid 층화 분할 (긍/부정 비율 유지)
    train_split, valid_split = train_test_split(
        train_df,
        test_size=valid_ratio,
        random_state=seed,
        stratify=train_df["label"]
    )

    train_split = train_split.reset_index(drop=True)
    valid_split = valid_split.reset_index(drop=True)

    train_split.to_csv(os.path.join(proc_dir, "train.csv"), index=False, encoding="utf-8-sig")
    valid_split.to_csv(os.path.join(proc_dir, "valid.csv"), index=False, encoding="utf-8-sig")
    test_df.to_csv(os.path.join(proc_dir, "test.csv"), index=False, encoding="utf-8-sig")

    print("\n=== 최종 분할 데이터 수 (보고서 기록용) ===")
    for split_name, split_df in [("Train", train_split), ("Valid", valid_split), ("Test", test_df)]:
        pos_cnt = (split_df["label"] == 1).sum()
        neg_cnt = (split_df["label"] == 0).sum()
        print(f"{split_name:5s}: 총 {len(split_df):7,d}개 (긍정 1: {pos_cnt:,} / 부정 0: {neg_cnt:,})")

if __name__ == "__main__":
    main()
