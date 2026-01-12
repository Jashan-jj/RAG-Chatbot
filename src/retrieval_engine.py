# Code for retrieval of most similar chunks.

from pymilvus import connections
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.text_utils import EMBEDDING_MODEL
from src.etl import MILVUS_HOST, MILVUS_PORT, create_milvus_collection
from utils.db_utils import CHUNKS_COLLECTION

connections.connect(alias="default", host=MILVUS_HOST, port=MILVUS_PORT)
collection = create_milvus_collection()

def search_chunks(query, top_k, article_type = "web"):
    """
    Retrieves top k semantically similar chunks in terms of their embeddings.

    Args:
        query: The input question given by the user
        top_k: Number of semantically similar chunks to retrieve
        article_type: Type of article (web/news)

    Returns:
        chunks: Top k semantically similar chunks
    """
    
    query_vector = EMBEDDING_MODEL.encode(query).tolist()
    search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
    collection.load()

    results = collection.search(
        data = [query_vector],
        anns_field = "embedding",
        param = search_params,
        limit = top_k,  
        output_fields = ["chunk_id"]
    )
    
    if not results or not results[0]:
        return []
    
    chunk_ids = [hit.entity.get("chunk_id") for hit in results[0]]
    chunks = list(CHUNKS_COLLECTION.find({"chunk_id": {"$in": chunk_ids}}))
    
    return chunks