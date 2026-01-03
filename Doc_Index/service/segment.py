import jieba
from typing import List, Dict

class SegmentService:

    def __init__(self):
        jieba.initialize()
        self.stopwords = self._load_stopwords()
    
    def _load_stopwords(self):
        stopwords = {
            '的', '了', '在', '是', '我', '有', '和', '就',
            '不', '人', '都', '一', '一个', '上', '也', '很',
            '到', '说', '要', '去', '你', '会', '着', '没有',
            '看', '好', '自己', '这'
        }
        return stopwords
    
    def predict(self, texts: List[str]) -> List[Dict[str, str]]:

        results = []
        
        for text in texts:
            words = jieba.cut(text, cut_all=False)
            filtered_words = [
                word for word in words 
                if word.strip() and word not in self.stopwords
            ]
            segmented_text = ' '.join(filtered_words)
            results.append({
                "text": text,
                "segment": segmented_text,
                "tokens": filtered_words,
                "token_count": len(filtered_words)
            })
        
        return results
    
    def predict_single(self, text: str) -> str:
        result = self.predict([text])
        return result[0]["segment"] if result else ""