import streamlit as st
from langchain_neo4j import Neo4jGraph
from dotenv import load_dotenv
import os

load_dotenv()


# Connect to Neo4j
graph = Neo4jGraph(
    url=os.getenv("NEO4J_URI"),
    username=os.getenv("NEO4J_USERNAME"),
    password=os.getenv("NEO4J_PASSWORD"),
)