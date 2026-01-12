# Code for all text-related functionalities.

from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import uuid
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from typing import List, Dict
import requests
from bs4 import BeautifulSoup
import re
from datetime import datetime

EMBEDDING_MODEL = SentenceTransformer("all-MiniLM-L6-v2", device='cpu') 

def chunk_text(document: str, document_id: str) -> List[Dict]:
    """
    Converts a document into chunks.

    Args:
        document (str): Document to be divided into chunks.
        document_id (str): A unique document_id associated with every document.

    Returns:
        chunk_schema (List[Dict]): List of dictionaries containing text chunks and metadata
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=10,
        separators=["\n", "."]
    )

    chunks = splitter.split_text(document)
    chunk_schema = [
        {
            "chunk_id": str(uuid.uuid4()),
            "chunk_text": chunk,
            "indexed": False,
            "metadata": {
                "document_id": document_id,
                "chunk_length": len(chunk),
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "method_used": None
            }
        }
        for chunk in chunks
    ]

    return chunk_schema

def clean_text(text):
    """
    Removes control characters and excessive whitespace from the given text.

    Args:
        text: Text to clean

    Returns:
        cleaned_text: Cleaned text
    """

    text = re.sub(r'[\x00-\x1F\x7F-\x9F]', ' ', text)  
    text = re.sub(r'\s+', ' ', text)  
    cleaned_text = text.strip()

    return cleaned_text

def get_html_text(url):
    """
    Extracts cleaned text and raw HTML from a given link.

    Args:
        url: Website link to extract clean text and raw HTML from

    Returns:
        cleaned_text: Cleaned text extracted from the link
        raw_html: Raw HTML extracted from the link
        title: Title of the link
    """

    try:
        response = requests.get(url, timeout=10)
        response.encoding = response.apparent_encoding  
        raw_html = response.text

        soup = BeautifulSoup(raw_html, "html.parser")

        for tag in soup(["script", "style", "noscript"]):
            tag.extract()

        title = soup.title.string.strip() if soup.title and soup.title.string else "No Title Found"

        main_content = soup.find("article") or soup.find("main")
        paragraphs = main_content.find_all("p") if main_content else soup.find_all("p")

        cleaned_text = " ".join(
            clean_text(p.get_text(separator=' ')) for p in paragraphs if p.get_text(strip=True)
        )

        return cleaned_text, raw_html, title

    except Exception as e:
        print(f"Error fetching URL: {e}")
        return "", "", "No title found"

def get_embedding(chunks: list[str], model_name = "all-MiniLM-L6-v2"):  
    """
    Generates an embedding vector for a given text block.

    Args:
        chunks (list[str]): Text to be converted into chunks and converted into embeddings
        model_name: Name of the model to be used for embedding

    Returns:
        embeddings: Embedding vector for the given text
    """
    
    embeddings = EMBEDDING_MODEL.encode(chunks, show_progress_bar=True, batch_size=128)

    return embeddings
