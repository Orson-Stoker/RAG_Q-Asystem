import re,jieba
from typing import List, Dict
from tqdm import tqdm



def _sliding_window_chunking(text,chunk_size=256,overlap=32):

    chunks = []
    text_len = len(text)
    start = 0
    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunk = text[start: end]
        chunks.append(chunk)
        if end == text_len:
            break
        start += (chunk_size - overlap)
    return chunks


def chunk_txt_file(doc_path,save_path):
    with open(doc_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        for line in lines:
            line=line.strip()
            chunks=_sliding_window_chunking(line)
            with open(save_path, 'a', encoding='utf-8') as f:
                for chunk in chunks:
                    f.write(chunk+'\n')

if __name__ == "__main__":
    doc_path=r"D:\StudyMaterial\PrincipleOfAI\HW3_RAG\Doc_Parsing\document.txt"
    save_path=r"D:\StudyMaterial\PrincipleOfAI\HW3_RAG\Doc_Parsing\doc_chunk.txt"
    chunk_txt_file(doc_path,save_path)

 