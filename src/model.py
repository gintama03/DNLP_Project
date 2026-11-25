import torch
import torch.nn as nn

class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, num_layers: int = 1, dropout: float = 0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )
        # 양방향(forward + backward) hidden state 결합 -> hidden_dim * 2
        self.fc = nn.Linear(hidden_dim * 2, 2)

    def forward(self, x):
        # x: (batch_size, max_len)
        emb = self.dropout(self.embedding(x))
        _, (h_n, _) = self.lstm(emb)
        # 마지막 레이어의 forward(-2)와 backward(-1) hidden state 연결
        hidden_cat = torch.cat((h_n[-2], h_n[-1]), dim=1)
        logits = self.fc(self.dropout(hidden_cat))
        return logits
