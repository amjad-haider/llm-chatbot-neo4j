import streamlit as st

from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings
# from langchain_groq import ChatGroq
import time
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
embeddings = OllamaEmbeddings(model=os.getenv("LLM_EMBEDDING_MODEL"))

# Neo4j connection details
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")



# Vector Index Configuration
INDEX_NAME = "moviePlots"
NODE_LABEL = "Movie"
TEXT_PROPERTY = "plot"
EMBEDDING_PROPERTY = "embedding"



# --- Helper Functions ---

def get_embedding_dimension(embedding_model):
    """Calculates the dimension of the embedding model by running a test query."""
    try:
        test_embedding = embedding_model.embed_query("test query")
        return len(test_embedding)
    except Exception as e:
        print(f"Could not determine embedding dimension: {e}")
        raise

def manage_neo4j_index(driver, required_dimension):
    """Checks the Neo4j index and creates or recreates it if necessary."""
    with driver.session() as session:
        try:
            # Check if the index exists and get its dimension
            result = session.run(
                "SHOW INDEXES YIELD name, options, type WHERE name = $name AND type = 'VECTOR'",
                name=INDEX_NAME,
            )
            record = result.single()
            
            if record is None:
                # --- Scenario 1: Index does not exist ---
                print(f"Index '{INDEX_NAME}' not found. Creating index with dimension {required_dimension}...")
                session.run(f"""
                CREATE VECTOR INDEX {INDEX_NAME} IF NOT EXISTS
                FOR (n:{NODE_LABEL}) ON (n.{EMBEDDING_PROPERTY})
                OPTIONS {{indexConfig: {{
                    `vector.dimensions`: {required_dimension},
                    `vector.similarity_function`: 'cosine'
                }}}}
                """)
                print("Index created successfully.")
            else:
                # --- Scenario 2 & 3: Index exists ---
                current_dimension = record["options"]["indexConfig"]["vector.dimensions"]
                print(f"Found existing index '{INDEX_NAME}' with dimension {current_dimension}.")
                
                if current_dimension != required_dimension:
                    # --- Scenario 2: Dimension mismatch ---
                    print(f"Dimension mismatch. Recreating index. Required: {required_dimension}, Found: {current_dimension}")
                    
                    # Drop the old index
                    print(f"Dropping index '{INDEX_NAME}'...")
                    session.run(f"DROP INDEX {INDEX_NAME}")
                    
                    # Wait a moment for the drop to complete
                    time.sleep(2) 
                    
                    # Create the new index
                    print(f"Creating new index '{INDEX_NAME}' with dimension {required_dimension}...")
                    session.run(f"""
                    CREATE VECTOR INDEX {INDEX_NAME}
                    FOR (n:{NODE_LABEL}) ON (n.{EMBEDDING_PROPERTY})
                    OPTIONS {{indexConfig: {{
                        `vector.dimensions`: {required_dimension},
                        `vector.similarity_function`: 'cosine'
                    }}}}
                    """)
                    print("Index recreated successfully.")
                else:
                    # --- Scenario 3: Dimensions match ---
                    print("✅ Index is up-to-date. No action needed.")

        except Exception as e:
            print(f"An error occurred during index management: {e}")
            raise
