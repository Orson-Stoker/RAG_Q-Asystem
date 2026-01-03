import numpy as np
from typing import List, Dict
import torch
from transformers import AutoTokenizer, AutoModel

class EmbeddingService:

    def __init__(self, model_name: str = "D:\StudyMaterial\PrincipleOfAI\HW3_RAG\Doc_Index\service\embedding_model"):
        self.model_name = model_name
        self.load_model()

    def load_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        if torch.cuda.is_available():
            self.model = self.model.cuda()
             
        self.model.eval()
 
    
    def predict(self, texts: List[str], max_length: int = 256) -> List[Dict[str, List[float]]]:  
        
        if not texts:
            return []
        
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        

        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state[:, 0, :]
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        
        embeddings = embeddings.cpu().numpy().tolist()
        
        results = []
        for i, text in enumerate(texts):
            results.append({
                "text": text,
                "embedding": embeddings[i],
                "dimension": len(embeddings[i])
            })
        
        return results
    
    def predict_single(self, text: str) -> List[float]:
        result = self.predict([text])
        return result[0]["embedding"] if result else []
    
