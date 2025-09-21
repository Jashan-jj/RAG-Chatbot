# Code for extraction and indexing purposes

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.text_utils import EMBEDDING_MODEL, get_html_text, get_embedding
from utils.db_utils import MILVUS_HOST, MILVUS_PORT, CHUNKS_COLLECTION, create_milvus_collection
from datetime import datetime
import uuid
import asyncio
from typing import List
from pymilvus import connections
from tqdm import tqdm

connections.connect(alias="default", host=MILVUS_HOST, port=MILVUS_PORT)

def extract_all_html(list_of_urls, doc_type):
    """
    Extracts cleaned text from a list of URLs.

    Args:
        list_of_urls: List of URLs from which cleaned data is to be extracted
        doc_type: Type/category to tag the documents with

    Returns:
        all_texts: List of document metadata and contents
    """

    all_texts = []

    for url in list_of_urls:
        print(f"Processing: {url}")
        cleaned_text, raw_html, title = get_html_text(url)
        all_texts.append({
            "document_id": str(uuid.uuid4()),
            "document_title": title,
            "document_text": cleaned_text,
            "document_raw_html": raw_html,
            "document_url": url,
            "document_type": doc_type,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })

    return all_texts

async def get_all_embeddings(list_of_text: List[str], batch_size: int = 128, model_name=EMBEDDING_MODEL) -> List[List[float]]:
    """
    Asynchronously creates embeddings of a list of texts.

    Args:
        list_of_text (List[str]): List of texts to be embedded
        batch_size: Size of the batch of texts to be embedded
        model_name: Model to be used for embedding

    Returns:
        results (List[List[float]]): List of embeddings
    """

    async def process_batch(batch):
        """
        Asynchronously processes each batch.

        Args:
            batch: Batch to be processed
        
        Returns:
            The result of 'get_embedding' executed asynchronously in a separate thread using 'asyncio.to_thread'
        """

        return await asyncio.to_thread(get_embedding, batch, model_name)

    batches = [list_of_text[i:i + batch_size] for i in range(0, len(list_of_text), batch_size)]

    results = []
    for batch in tqdm(batches, desc="Batches"):
        embeddings = await process_batch(batch)
        results.extend(embeddings)

    return results

async def index_documents():
    """
    Asynchronously indexes all unindexed documents in the database.

    Returns:
        None
    """
    
    print("\nFetching unindexed chunks...")
    unindexed_chunks = list(CHUNKS_COLLECTION.find({"indexed": False}))
    if not unindexed_chunks:
        print("No unindexed chunks found.")
        return

    print(f"Found {len(unindexed_chunks)} chunks.")
    texts = [chunk["chunk_text"] for chunk in unindexed_chunks]
    chunk_ids = [chunk["chunk_id"] for chunk in unindexed_chunks]

    print("\nGenerating embeddings...")
    embeddings = await get_all_embeddings(texts, batch_size=128)
    print(f"Generated {len(embeddings)} embeddings")

    print("\nCreating Milvus collection...")
    collection = create_milvus_collection()
    print("Collection created.")

    print("\nInserting into Milvus...")
    collection.insert([ 
        embeddings,
        chunk_ids
    ])
    print("Insertion done.")

    print("\nUpdating MongoDB flags...")
    for chunk in unindexed_chunks:
        CHUNKS_COLLECTION.update_one(
            {"_id": chunk["_id"]},
            {"$set": {"indexed": True}}
        )
    print("Updation done.")

    print("\nFlushing collection...")
    collection.flush()
    print("Milvus index updated and saved locally.")