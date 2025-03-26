import os
import requests
from pinecone import Pinecone  
from dotenv import load_dotenv
from pinecone_indexing import get_huggingface_embedding  

# Load environment variables
load_dotenv(dotenv_path=".env")

# Initialize Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))  

# Define index name
INDEX_NAME = os.getenv("PINECONE_INDEX")

# Ensure the index exists
if INDEX_NAME not in pc.list_indexes().names():
    raise ValueError(f"Error: The Pinecone index '{INDEX_NAME}' does not exist. Please create it first.")

# Connect to the existing index
index = pc.Index(INDEX_NAME)

# Claude API Configuration
CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY")
CLAUDE_API_URL = "https://api.anthropic.com/v1/messages"

def retrieve_relevant_chunks(question, top_k=5):
    """Retrieve the most relevant chunks from Pinecone based on the Markdown content."""
    query_embedding = get_huggingface_embedding(question)  

    search_results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True
    )

    retrieved_texts = [match["metadata"]["text"] for match in search_results["matches"]]
    return "\n".join(retrieved_texts)

def answer_question_claude(question):
    """Retrieves relevant document chunks and asks Claude for an answer."""
    try:
        # Retrieve relevant text from Pinecone
        context = retrieve_relevant_chunks(question)

        if not context:
            return "No relevant information found in the indexed Markdown document."

        headers = {
            "x-api-key": CLAUDE_API_KEY,  
            "anthropic-version": "2023-06-01",  
            "Content-Type": "application/json"
        }
        payload = {
            "model": "claude-3-opus-20240229",  
            "max_tokens": 300,  
            "system": "You are an AI that answers questions based on the given document.",  
            "messages": [
                {"role": "user", "content": f"Context:\n{context}\n\nAnswer the following question based ONLY on the document above:\n{question}"}
            ],
            "temperature": 0.7  
        }

        response = requests.post(CLAUDE_API_URL, headers=headers, json=payload)

        if response.status_code == 200:
            return response.json()["content"]
        else:
            return f"Claude API Error: {response.json()}"

    except Exception as e:
        return f"Claude RAG Failed: {str(e)}"