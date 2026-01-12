# Code for all database-related functionalities.

from pymongo import MongoClient  
from dotenv import load_dotenv 
import os
from pymilvus import Collection, utility, FieldSchema, CollectionSchema, DataType

load_dotenv()

MILVUS_COLLECTION = "documents_chunks"
MILVUS_HOST = "localhost"
MILVUS_PORT = "19530"
EMBEDDING_DIM = 384

ATLAS_URI = os.getenv("ATLAS_URI")
DB_NAME = "B14_Chatbot_News_Explorer" 
client = MongoClient(ATLAS_URI)      
DB = client[DB_NAME] 
DOCS_COLLECTION = DB["documents_main"]
CHUNKS_COLLECTION = DB["documents_chunks"]

print("Connections established successfully.")

def insert_in_db_document(document):
    """
    Inserts each document into the MongoDB Collection (documents_main).

    Args:
        document: Single document to be inserted

    Returns:
        None
    """

    collection = DOCS_COLLECTION
    try:
        result = collection.insert_one(document)
        print("Inserted document with ID: ", result.inserted_id)    

    except Exception as e:
        print("Error inserting document: ", e)
        return None

def insert_all_documents(list_of_documents):
    """
    Dumps all documents in MongoDB Atlas collection (documents_main).

    Args:
        list_of_documents: List of documents to be dumped.

    Returns:
        None
    """

    for document in list_of_documents:
        insert_in_db_document(document)
    
    return None

def insert_in_db_chunks(chunk):
    """
    Inserts a single chunk into the database (documents_chunks).

    Args:
        chunk: Single chunk to be inserted

    Returns:
        None
    """

    collection = CHUNKS_COLLECTION
    if "_id" in chunk:
        del chunk["_id"]
    try:
        result = collection.insert_one(chunk)
        print("Inserted chunk with ID: ", result.inserted_id)
    except Exception as e:
        print("Error inserting chunk: ", e)
    return None
    
def insert_all_chunks(list_of_chunks):
    """
    Dumps all chunks in MongoDB Atlas collection (documents_chunks).

    Args:
        list_of_chunks: List of chunks to be dumped.

    Returns:
        None
    """

    for chunk in list_of_chunks:
        if isinstance(chunk, list):
            if len(chunk) == 0:
                print("Skipping empty chunk list.")
                continue
            elif len(chunk) == 1 and isinstance(chunk[0], dict):
                insert_in_db_chunks(chunk[0])
            else:
                print("Skipping invalid chunk (list with length !=1 or non-dict).")
        elif isinstance(chunk, dict):
            insert_in_db_chunks(chunk)
        else:
            print("Skipping invalid chunk (not dict or list).")

    return None

def clear_all_chunks(collection):
    """
    Clears all the chunks from the MongoDB Atlas database (to be used only when necessary).

    Args:
        collection: Collection to clear chunks from

    Returns:
        None
    """

    try: 
        result = collection.delete_many({})
        print(f"Deleted {result.deleted_count} chunks from the collection.")
    except Exception as e:
        print(f"Error clearing chunks from MongoDB collection: {e}")

def clear_all_docs(collection):
    """
    Clears all the documents from the MongoDB Atlas database (to be used only when necessary).

    Args:
        collection: Collection to clear documents from

    Returns:
        None
    """

    try:
        result = collection.delete_many({})
        print(f"Deleted {result.deleted_count} documents from the collection.")
    except Exception as e:
        print(f"Error clearing documents from MongoDB collection: {e}")

def create_milvus_collection():
    """
    Creates Milvus vector database collection.

    Returns:
        collection: Milvus vector database collection
    """

    if not utility.has_collection(MILVUS_COLLECTION):
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM),
            FieldSchema(name="chunk_id", dtype=DataType.VARCHAR, max_length=100)
        ]
        schema = CollectionSchema(fields, description="Document chunks embedding")
        collection = Collection(name=MILVUS_COLLECTION, schema=schema)
        collection.create_index(field_name="embedding", index_params={
            "metric_type": "L2",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 128}
        })
    else:
        collection = Collection(MILVUS_COLLECTION)

    return collection

def delete_milvus_collection(collection_name: str):
    """
    Deletes Milvus vector database collection.

    Args:
        collection_name (str): Name of the Milvus connection to be deleted

    Returns:
        None
    """
    
    try:
        if utility.has_collection(collection_name):
            utility.drop_collection(collection_name)
            print("Milvus collection deleted successfully.")
        else:
            print(f"Collection '{collection_name}' does not exist.")
    except Exception as e:
        print(f"Error deleting collection: {e}")
