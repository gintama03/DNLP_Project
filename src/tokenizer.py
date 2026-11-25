import re
from kiwipiepy import Kiwi

class KoreanTokenizer:
    def __init__(self, normalize: bool = False):
        self.kiwi = Kiwi()
        self.normalize = normalize

    def clean_text(self, text: str) -> str:
        text = str(text).strip()
        if self.normalize:
            # 구어체 노이즈 정규화 (Model C 비교 실험용): 3회 이상 반복되는 자모/문자 축약 (예: ㅋㅋㅋㅋ -> ㅋㅋ)
            text = re.sub(r"(.)\1{2,}", r"\1\1", text)
            # URL 및 연속 공백 정리
            text = re.sub(r"http\S+|www\S+", " <URL> ", text)
            text = re.sub(r"\s+", " ", text).strip()
        return text

    def tokenize(self, text: str) -> list[str]:
        text = self.clean_text(text)
        if not text:
            return []
        # Kiwi 형태소 분석기에서 표면형(form) 추출
        tokens = [t.form for t in self.kiwi.tokenize(text)]
        return tokens
