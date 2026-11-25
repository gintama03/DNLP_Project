import json
import yaml
import torch
from tokenizer import KoreanTokenizer
from model import BiLSTMClassifier

def predict_sentences(sentences, model, tokenizer, vocab, max_len, device):
    model.eval()
    unk_id = vocab["<UNK>"]
    pad_id = vocab["<PAD>"]
    results = []

    with torch.no_grad():
        for text in sentences:
            tokens = tokenizer.tokenize(text)
            ids = [vocab.get(t, unk_id) for t in tokens][:max_len]
            if len(ids) < max_len:
                ids += [pad_id] * (max_len - len(ids))
            x = torch.tensor([ids], dtype=torch.long).to(device)
            logits = model(x)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            pred = int(probs[1] >= 0.5)
            conf = probs[1] if pred == 1 else probs[0]
            results.append((text, tokens, pred, conf))
    return results

def main():
    with open("config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(cfg["vocab"]["vocab_path"], "r", encoding="utf-8") as f:
        vocab = json.load(f)

    model = BiLSTMClassifier(
        vocab_size=len(vocab),
        embed_dim=cfg["model"]["embed_dim"],
        hidden_dim=cfg["model"]["hidden_dim"],
        num_layers=cfg["model"]["num_layers"],
        dropout=cfg["model"]["dropout"]
    ).to(device)
    model.load_state_dict(torch.load("outputs/model_b_bilstm.pt", map_location=device))

    tokenizer = KoreanTokenizer(normalize=False)

    # 직접 작성한 새로운 입력 5개 (긍정, 부정, 복합/반어 등 다양한 케이스)
    new_reviews = [
        "배우들 연기력이 미쳤고 연출도 여운이 길게 남는 인생 영화입니다.",
        "시간이랑 돈이 너무 아깝다. 스토리 개연성이 하나도 없고 지루함의 끝판왕.",
        "초반에는 조금 지루했는데 후반부 반전이 소름 돋아서 별점 10점 줍니다!",
        "기대를 너무 많이 했는지 예고편이 전부인 느낌이라 솔직히 실망스러웠어요.",
        "CG는 화려한데 내용이 산으로 가서 보다가 중간에 졸았습니다."
    ]

    preds = predict_sentences(new_reviews, model, tokenizer, vocab, cfg["model"]["max_len"], device)

    lines = ["=== 직접 작성한 새로운 리뷰 5개 추론 결과 (Model B) ===\n"]
    for i, (text, tokens, pred, conf) in enumerate(preds, start=1):
        label_str = "긍정 (1)" if pred == 1 else "부정 (0)"
        block = (
            f"[입력 {i}] {text}\n"
            f"  - 토큰: {tokens}\n"
            f"  - 예측: {label_str} (확신도: {conf*100:.2f}%)\n"
        )
        lines.append(block)

    output_text = "\n".join(lines)
    print(output_text)
    with open("outputs/demo_predictions.txt", "w", encoding="utf-8") as f:
        f.write(output_text)

if __name__ == "__main__":
    main()
