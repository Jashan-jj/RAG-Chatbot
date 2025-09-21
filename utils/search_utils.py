# Code to get the associated links using Serper API.

import requests
from dotenv import load_dotenv
import os
import traceback as tr

load_dotenv() 

web_url = os.getenv("SERPER_WEB")
news_url = os.getenv("SERPER_NEWS")
api_key = os.getenv("SERPER_API_KEY")

headers = {
    "X-API-KEY": api_key, 
    "Content-Type": "application/json"
}

def search_web(query:str, nb_docs = 10):
    """
    Performs a web search using Serper API

    Args:
        query (str): Question input given by the user
        nb_docs: Number of documents

    Returns:
        links: Search results for the query
        "web" (str): URL type
    """

    data = {
        "q": query,
        "num": nb_docs
    }

    try:
        response = requests.post(url=web_url, headers=headers, json=data).json()['organic']

        links = []
        for i in response:
            links.append(i['link'])
    except:
        print("Can not generate URLs as ", tr.format_exc())

    return links, "web"

def search_news(query:str, nb_docs = 10):
    """
    Performs a news search using Serper API

    Args:
        query (str): Question input given by the user
        nb_docs: Number of documents

    Returns:
        links: Search results for the query
        "news" (str): URL type
    """

    data = {
        "q": query,
        "num": nb_docs
    }

    try:
        response = requests.post(url=news_url, headers=headers, json=data).json()['news']

        links = []
        for i in response:
            links.append(i['link'])
    except:
        print("Can not generate URLs as ", tr.format_exc())

    return links, "news"
