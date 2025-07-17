import streamlit as st

from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings
# from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

load_dotenv()

# Create the LLM
llm = ChatOllama(
    model=os.getenv("LLM_MODEL"),
    temperature=0,
)
# llm = ChatGroq(
#     model="meta-llama/llama-4-scout-17b-16e-instruct",
#     temperature=0,
# )

# Create the Embedding model
embeddings = OllamaEmbeddings(model=os.getenv("LLM_MODEL"))