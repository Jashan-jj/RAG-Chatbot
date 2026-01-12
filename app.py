# Main program

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.search_utils import search_web, search_news
from utils.db_utils import insert_all_chunks, insert_all_documents
from utils.text_utils import chunk_text
# from utils.user_interface import launch_chatbot
from src.etl import extract_all_html, index_documents
import asyncio

async def ingestion(query, nb_docs = 10):
    """
    Inserts chunks, documents, and embeddings into respective databases followed by indexing them appropriately.
    Primarily used for the purpose of training.

    Args:
        query: Question input given by the user
        nb_docs: Number of documents to get information from

    Returns:
        None
    """

    print("\nGetting results from Serper API...")
    results1, doc_type1 = search_web(query, nb_docs)
    results2, doc_type2 = search_news(query, nb_docs)
    print("Successfully got results from Serper API.")

    print("\nScraping results got from Serper API...")
    all_texts = extract_all_html(results1, doc_type1)
    all_texts.extend(extract_all_html(results2, doc_type2))
    print("Successfully scraped results got from Serper API.")
    
    print("\nStoring all documents...")
    insert_all_documents(all_texts)
    print("Successfully stored all documents.")

    print("\nChunking documents...")
    all_chunks_schema = []
    for text in all_texts:
        chunks = chunk_text(text["document_text"], text["document_id"])                                        
        all_chunks_schema.extend(chunks)
    print("Successfully chunked documents.")

    print("\nInserting chunks in MongoDB Atlas...")
    insert_all_chunks(all_chunks_schema) 
    print("Successfully inserted chunks in MongoDB Atlas.")

    print("\nIndexing documents...")
    await index_documents()
    print("\n\nAll done")

# if __name__ == "__main__":
#     launch_chatbot()
    
    

    
    




    


