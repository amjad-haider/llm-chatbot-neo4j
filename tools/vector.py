import streamlit as st
from llm import llm, embeddings,manage_neo4j_index,get_embedding_dimension
from graph import graph
from langchain_neo4j import Neo4jVector
from neo4j import GraphDatabase
from dotenv import load_dotenv
import os

load_dotenv()
# Create the Neo4jVector

# Sanity Check

try:
        # 1. Determine the required dimension from the embedding model
        required_dim = get_embedding_dimension(embeddings)
        print(f"Using embedding model '{os.getenv("LLM_EMBEDDING_MODEL")}' which requires dimension: {required_dim}")
        
        # 2. Connect to Neo4j and manage the index
        with GraphDatabase.driver(os.getenv("NEO4J_URI"), auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD"))) as neo4j_driver:
            manage_neo4j_index(neo4j_driver, required_dim)
            
        # 3. Safely proceed to create your LangChain retriever
        print("\nIndex management complete. Proceeding with LangChain setup...")


except Exception as e:
    print(f"\nFailed to initialize the vector database: {e}")


neo4jvector = Neo4jVector.from_existing_index(
    embeddings,                              # (1)
    graph=graph,                             # (2)
    index_name="moviePlots",                 # (3)
    node_label="Movie",                      # (4)
    text_node_property="plot",               # (5)
    embedding_node_property="plotEmbedding", # (6)
    retrieval_query="""
RETURN
    node.plot AS text,
    score,
    {
        title: node.title,
        directors: [ (person)-[:DIRECTED]->(node) | person.name ],
        actors: [ (person)-[r:ACTED_IN]->(node) | [person.name, r.role] ],
        tmdbId: node.tmdbId,
        source: 'https://www.themoviedb.org/movie/'+ node.tmdbId
    } AS metadata
"""
)

# Create the retriever
retriever = neo4jvector.as_retriever()
# Create the prompt
from langchain_core.prompts import ChatPromptTemplate

instructions = (
    "Use the given context to answer the question."
    "If you don't know the answer, say you don't know."
    "Context: {context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", instructions),
        ("human", "{input}"),
    ]
)
# Create the chain 
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

question_answer_chain = create_stuff_documents_chain(llm, prompt)
plot_retriever = create_retrieval_chain(
    retriever, 
    question_answer_chain
)
# Create a function to call the chain
def get_movie_plot(input):
    return plot_retriever.invoke({"input": input})
