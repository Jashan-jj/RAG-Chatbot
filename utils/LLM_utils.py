# Code for all Large Language Model (LLM)-related functionalities.
import os
import sys
import asyncio
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dotenv import load_dotenv
from src.retrieval_engine import search_chunks
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from app import ingestion

load_dotenv()

google_api_key = os.getenv("GOOGLE_API_KEY")
os.environ["GOOGLE_API_KEY"] = google_api_key

llm = ChatGoogleGenerativeAI(model = "gemini-2.5-flash")

def rank_chunks(query, chunks):
    """
    Ranks chunks from most relevant to least relevant.

    Args:
        query: Question input given by the user
        chunks: Chunks to be ranked

    Returns:
        reranked_chunks: Chunks after re-ranking
    """

    chunk_texts = [chunk["chunk_text"] for chunk in chunks]

    prompt_template = PromptTemplate(
        input_variables=["query", "chunks"],
        template="""
        Given the query: "{query}"

        Rank the following chunks from most relevant to least relevant.
        Chunks:
        {chunks}

        Return only the list of chunk numbers in ranked order e.g., "2, 1, 3, 5, 4"
        """
    )
    
    formatted_chunks = "\n".join([f"{i+1}. {text}" for i, text in enumerate(chunk_texts)])
    prompt = prompt_template.format(query=query, chunks=formatted_chunks)

    response = llm.invoke(prompt)

    ranked_order = response.content.strip()
    indices = [int(x.strip()) - 1 for x in ranked_order.split(",") if x.strip().isdigit()]
    reranked_chunks = [chunks[i]["chunk_text"] for i in indices if 0 <= i < len(chunks)]

    return reranked_chunks

memory = ConversationBufferMemory(
    memory_key="chat_history",
    input_key="query",         
    return_messages=True
    )

def llm_call(query, support_material):
    """
    Invokes an LLM call.

    Args:
        query: Question input given by the user
        support_material: Text to provide context

    Returns:
        response: Response to the query by the LLM
    """

    prompt = PromptTemplate(
        input_variables=["query", "support_material", "chat_history"],
        template="""
        This is the supporting material.
        {support_material}
        The is a conversation between a helpful assistant and a user.
        {chat_history}
        User: {query}
        Assistant:"""
    )

    conversation_chain = LLMChain(
        llm=llm,
        prompt=prompt,
        memory=memory
    )

    response = conversation_chain.run({
    "query": query,
    "support_material": support_material
    })
    
    return response

async def llm_response(query):
    """
    Makes the final call to the LLM to elicit a response.

    Args:
        query: Question input given by the user

    Returns:
        response: Response to the query by the LLM
    """
    await ingestion(query)
    top_chunks = search_chunks(query, 5, "web")
    reranked = rank_chunks(query, top_chunks)
    response = llm_call(query, reranked)
    return response
