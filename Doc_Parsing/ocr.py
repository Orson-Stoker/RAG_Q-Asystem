from paddleocr import PaddleOCR
from tqdm import tqdm
import os

def Document_Parsing(doc_path,save_path):

    ocr = PaddleOCR(
        use_textline_orientation=False,  
        text_det_thresh=0.3,           
        text_det_box_thresh=0.5,     
        det_limit_side_len=1280,      
        rec_batch_num=6,            
        lang='ch',
    )

    results = ocr.predict(doc_path)

    for result in results:
        if result and 'rec_texts' in result:
            text_lines = result['rec_texts']

            text="".join(text_lines)+"\n"
            
        with open(save_path, 'a', encoding='utf-8') as f:
            f.writelines(text) 

def Document_Dir_Parsing(doc_dir,save_path):
    doc_list=os.listdir(doc_dir)
    for doc in tqdm(doc_list,desc=f"OCR processing"):
        print(f"\nExtract information from {doc}")
        doc_path=os.path.join(doc_dir,doc)
        Document_Parsing(doc_path,save_path)


if __name__ == "__main__":
    save_path="D:\StudyMaterial\PrincipleOfAI\HW3_RAG\Doc_Parsing\document.txt"
    doc_dir=r"Doc_Parsing\document_dir"
    Document_Dir_Parsing(doc_dir,save_path)

