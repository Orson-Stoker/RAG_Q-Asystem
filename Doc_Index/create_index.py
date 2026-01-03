from opensearchpy import OpenSearch
from .service.embedding import EmbeddingService
from .service.segment import SegmentService 
from typing import List, Dict, Any
from tqdm import tqdm
import hashlib, json

class OpenSearch_Manager:
    def __init__(self):
        print("Logging into Opensearch...")
        self.client = OpenSearch(
            hosts=[{'host': 'localhost', 'port': 9200}], 
            http_auth=('admin', 'admin'),  
            use_ssl=False, 
            verify_certs=False, 
            ssl_show_warn=False
        )
        self.check_connection()
        print("Loading Service Model...")

        self.embedding_service = EmbeddingService()
        self.segment_service = SegmentService()

    def check_connection(self):

        try:
            if self.client.ping():
                print("✅ OpenSearch Successful Connection")
                return True
            else:
                print("❌ OpenSearch Failed Connection")
                return False
        except Exception as e:
            print(f"❌ Wrong Conneciton: {e}")
            return False

    def create_index(self, index_config=r"Doc_Index\index_config.json", index_name="my_knowledge"):   
        with open(index_config, 'r', encoding='utf-8') as f:
            index_body = json.load(f)

        if self.client.indices.exists(index=index_name):
            print(f"⚠️  index {index_name} existed, deleting...")
            self.client.indices.delete(index=index_name)
        
        response = self.client.indices.create(
            index=index_name,
            body=index_body
        )
        
        print(f"✅ Index {index_name} has been created successfully")
        return response


    def write_doc(self, doc_path):
        with open(doc_path, "r", encoding='utf-8') as file:
            lines = file.readlines()
            for line in tqdm(lines,desc="Writing into Opensearch Library...",unit="line"):
                body = self.create_doc_body(line, str(self.create_id(line)))
                
                self.client.index(
                    index="my_knowledge",   
                    id=body["id"],       
                    body=body,          
                    refresh=True        
                )
            print(f"{len(lines)} lines of data in total have been logged.")

    def create_doc_body(self, knowledge, id):
        embedding_result = self.embedding_service.predict_single(knowledge)
        segment_result = self.segment_service.predict_single(knowledge)

        body = {
            "id": id,
            "knowledge": str(knowledge),
            "segment": segment_result,
            "vector": embedding_result
        }
        return body

    def create_id(self, text):
        def cal_hash(text):
            md5 = hashlib.md5(bytes(text, encoding="utf8")).hexdigest()
            return int(md5, 16)
        return cal_hash(text)
    
    def rrf_hybrid_search(self, query_text: str, top_k: int = 10, 
                          vector_top_n: int = 100, text_top_n: int = 100,
                          rrf_constant: int = 60, index_name: str = "my_knowledge") -> List[Dict[str, Any]]:
        
        
        vector_results = self.cos_vector_search_with_rank(query_text, vector_top_n, index_name)
        bm25_results = self.bm25_text_search_with_rank(query_text, text_top_n, index_name)
        
   
        print(f"Vector Retrieval: {len(vector_results)} results")
        print(f"Text Retrieval:{len(bm25_results)} results")

        rrf_results = self._rrf_fusion(vector_results, bm25_results, rrf_constant, top_k)
        return rrf_results

    def cos_vector_search_with_rank(self, query_text: str, top_n: int, index_name: str) -> Dict[str, Dict]:
        try:
            query_vector = self.embedding_service.predict_single(query_text)
            
            search_body = {
                "size": top_n,
                "query": {
                    "knn": {
                        "vector": {
                            "vector": query_vector,
                            "k": top_n
                        }
                    }
                },
                "_source": ["knowledge"],
                "sort": [
                    {
                        "_score": {"order": "desc"}
                    }
                ]
            }
            
            response = self.client.search(index=index_name, body=search_body)
            
            results = {}
            for rank, hit in enumerate(response['hits']['hits'], 1):
                doc_id = hit['_id']
                score = hit.get('_score', 0)
          
                results[doc_id] = {
                    'vector_score': score,
                    'rank': rank,
                    'content': hit['_source'].get('knowledge', '')
                    
                }
            
            return results
        except Exception as e:
            print(f"Vector Retrieval Wrong: {e}")
            return {}

    def bm25_text_search_with_rank(self, query_text: str, top_n: int, index_name: str) -> Dict[str, Dict]:
        try:
            segmented_query = self.segment_service.predict_single(query_text)

            search_body = {
                "size": top_n,
                "query": {
                    "match": {
                        "segment": 
                           segmented_query
                    }
                },
                "_source": ["knowledge"],
                "explain": True,  
                "sort": [
                    {
                        "_score": {"order": "desc"}
                    }
                ]
            }
            
            response = self.client.search(index=index_name, body=search_body)   
            results = {}
            for rank, hit in enumerate(response['hits']['hits'], 1):
                doc_id = hit['_id']
                score = hit.get('_score', 0)
                
                results[doc_id] = {
                    'bm25_score': score,
                    'rank': rank,
                    'content': hit['_source'].get('knowledge', '')
                }
            
            return results
        except Exception as e:
            print(f"Text Retrieval Wrong: {e}")
            return {}

    def _rrf_fusion(self, vector_results: Dict[str, Dict], bm25_results: Dict[str, Dict],
                    rrf_constant: int, top_k: int) -> List[Dict[str, Any]]:
        
        all_docs = set(vector_results.keys()) | set(bm25_results.keys())

        rrf_scores = {}
        
        for doc_id in all_docs:
            vector_rank = vector_results.get(doc_id, {}).get('rank', float('inf'))
            bm25_rank = bm25_results.get(doc_id, {}).get('rank', float('inf'))
            
            vector_score = vector_results.get(doc_id, {}).get('vector_score', 0)
            bm25_score = bm25_results.get(doc_id, {}).get('bm25_score', 0)
            
            rrf_score = 0
            
            if vector_rank != float('inf'):
                rrf_score += 1.0 / (rrf_constant + vector_rank)
            
            if bm25_rank != float('inf'):
                rrf_score += 1.0 / (rrf_constant + bm25_rank)
            
            if rrf_score > 0:
                rrf_scores[doc_id] = {
                    'rrf_score': rrf_score,
                    'vector_score': vector_score,
                    'bm25_score': bm25_score,
                    'vector_rank': vector_rank if vector_rank != float('inf') else None,
                    'bm25_rank': bm25_rank if bm25_rank != float('inf') else None,
                    'content': vector_results.get(doc_id, {}).get('content') or 
                              bm25_results.get(doc_id, {}).get('content', '')
                }
        

        sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1]['rrf_score'], reverse=True)

        final_results = []
        for i, (doc_id, scores) in enumerate(sorted_results[:top_k]):
            
            final_results.append({
                'rank': i + 1,
                'id': doc_id,
                'rrf_score': scores['rrf_score'],
                'vector_score': scores['vector_score'],
                'bm25_score': scores['bm25_score'],
                'vector_rank': scores['vector_rank'],
                'bm25_rank': scores['bm25_rank'],
                'content': scores['content'],
            })
        
        return final_results
    
    
    def data_check(self, size):
        response = self.client.search(
            index="my_knowledge",
            body={
                "query": {"match_all": {}},
                "size": size 
            }
        )

        for i, hit in enumerate(response['hits']['hits'], 1):
            print(f"\n📄 document {i}:")
            print(f"   ID: {hit['_id']}")
            print(f"   content: {hit['_source']['knowledge'][:100]}...")  
            print(f"   vector: {hit['_source']['vector']}")
            print(f"   segment: {hit['_source']['segment']}")

    def query_opensearch(self,query):
        res=self.rrf_hybrid_search(query)
        for item in res:
            print(f"\nRank {item['rank']} ")
            print(f"  Doc ID: {item['id']}")
            print(f"  RRF score: {item['rrf_score']:.6f}")
            print(f"  vector score: {item['vector_score']:.4f}")
            print(f"  BM25 score: {item['bm25_score']:.4f}")
            print(f"  content: {item['content']}...")

if __name__ == "__main__":
    doc_path = r"D:\StudyMaterial\PrincipleOfAI\HW3_RAG\Doc_Parsing\doc_chunk.txt"
    manager = OpenSearch_Manager()
    manager.create_index()
    manager.write_doc(doc_path)

