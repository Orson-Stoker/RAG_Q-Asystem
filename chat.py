import openai,sys,os
from typing import List, Dict, Optional
from dataclasses import dataclass
from Doc_Index.create_index import *
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch

sys.path.append('Doc_Index')
sys.path.append('Doc_Parsing')

@dataclass
class RetrievedDoc:
    id:      int
    content: str
    score: float  


class RAGResponseGenerator_API:
    def __init__(self, api_key: str="XXXXXXXXXXXXXX", base_url: str = "https://api.deepseek.com"):

        self.client = openai.OpenAI(
            api_key=api_key,
            base_url=base_url
        )
        self.manager=OpenSearch_Manager()
    
    def format_context(self, retrieved_docs: List[RetrievedDoc], 
                      max_context_length: int = 1000) -> str:

        context_parts = []
        current_length = 0
        
        for i, doc in enumerate(retrieved_docs):
            doc_text = f"[文档{i+1} - 相关性:{doc.score:.3f}]\n{doc.content}\n\n"
            
            if current_length + len(doc_text) > max_context_length:
                break
                
            context_parts.append(doc_text)
            current_length += len(doc_text)
        
        return "".join(context_parts)
    
    def generate_with_context(self, 
                             query: str,
                             system_prompt: Optional[str] = None,
                             model: str = "deepseek-chat",
                             temperature: float = 0.3,  
                             max_tokens: int = 1500,
                            ) -> Dict:
        retrieved_docs=[]
        results=self.manager.rrf_hybrid_search(query_text=query)
        for item in results:
            doc = RetrievedDoc(
            id=item['id'],
            content=item['content'],
            score=item['rrf_score'])
            retrieved_docs.append(doc)
        
        
        context = self.format_context(retrieved_docs)
     
    
        if system_prompt is None:
            system_prompt = """你是一个专业的RAG助手。请基于提供的参考文档来回答用户问题。
            
            请遵守以下规则：
            1. 严格基于参考文档的内容进行回答
            2. 如果参考文档中没有相关信息，请如实告知无法回答
            3. 可以整合多个文档的信息，但要注明来源
            4. 回答要简洁、准确、专业
            5. 引用文档时注明文档编号，如[文档1]
            
            参考文档：
            """
        
        full_system_prompt = f"{system_prompt}\n\n{context}"
        
        
        messages = [
            {"role": "system", "content": full_system_prompt},
            {"role": "user", "content": query}
        ]
        
      
        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        
     
        reply = response.choices[0].message.content
        
        return {
            "response": reply,
            "model": model,
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            },
            "context_docs_count": len(retrieved_docs),
            "doc_scores": [doc.score for doc in retrieved_docs],
            "context":context
        }
    
    def chat(self):
        while 1:
            query=str(input("请输入你的问题："))
            result=self.generate_with_context(query)
            print("-"*20)
            print(f"回答：{result['response']}")
            print("-"*20)
            print(f"依据：\n{result['context']}")
    

class RAGResponseGenerator_Local:
    def __init__(self,model_path="D:\StudyMaterial\PrincipleOfAI\HW3_RAG\Chat_local_model"):

        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        self.manager=OpenSearch_Manager()
    
    def format_context(self, retrieved_docs: List[RetrievedDoc], 
                      max_context_length: int = 1000) -> str:

        context_parts = []
        current_length = 0
        
        for i, doc in enumerate(retrieved_docs):
            doc_text = f"[文档{i+1} - 相关性:{doc.score:.3f}]\n{doc.content}\n\n"
            
            if current_length + len(doc_text) > max_context_length:
                break
                
            context_parts.append(doc_text)
            current_length += len(doc_text)
        
        return "".join(context_parts)
    
    def generate_with_context(self, 
                             query: str,
                             system_prompt: Optional[str] = None,
                             model: str = "deepseek-chat",
                             temperature: float = 0.7,  
                             max_tokens: int = 512,
                            ) -> Dict:
        retrieved_docs=[]
        results=self.manager.rrf_hybrid_search(query_text=query)
        for item in results:
            doc = RetrievedDoc(
            id=item['id'],
            content=item['content'],
            score=item['rrf_score'])
            retrieved_docs.append(doc)
        
        
        context = self.format_context(retrieved_docs)
     
    
        if system_prompt is None:
            system_prompt = """你是一个专业的RAG助手。请基于提供的参考文档来回答用户问题。
            
            请遵守以下规则：
            1. 严格基于参考文档的内容进行回答
            2. 如果参考文档中没有相关信息，请如实告知无法回答
            3. 可以整合多个文档的信息，但要注明来源
            4. 回答要简洁、准确、专业
            5. 引用文档时注明文档编号，如[文档1]
            
            参考文档：
            """
        
        full_system_prompt = f"{system_prompt}\n\n{context}"
        

        messages = [
            {"role": "user", "content": full_system_prompt},
            {"role": "user", "content": query}
        ]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=0.9
            )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
     
        reply = response
        
        return {
            "response": reply,
            "model": model,
            "context_docs_count": len(retrieved_docs),
            "doc_scores": [doc.score for doc in retrieved_docs],
            "context":context
        }
    
    def chat(self):
        while 1:
            query=str(input("请输入你的问题："))
            result=self.generate_with_context(query)
            print("-"*20)
            print(f"回答：{result['response']}")
            print("-"*20)
            print(f"依据：\n{result['context']}")


if __name__ == "__main__":
    
    robot=RAGResponseGenerator_Local()
    #robot=RAGResponseGenerator_API()
    robot.chat()