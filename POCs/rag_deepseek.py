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

# DeepSeek API Configuration
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"

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

def answer_question_deepseek(question):
    """Retrieves relevant document chunks and asks DeepSeek for an answer."""
    try:
        # Retrieve relevant text from Pinecone
        context = retrieve_relevant_chunks(question)

        if not context:
            return "No relevant information found in the indexed Markdown document."

        headers = {
            "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "deepseek-chat",
            "messages": [
                {"role": "system", "content": "You are an AI that answers questions based on the given document."},
                {"role": "user", "content": f"Context:\n{context}\n\nAnswer the following question based ONLY on the document above:\n{question}"}
            ],
            "temperature": 0.7
        }

        response = requests.post(DEEPSEEK_API_URL, headers=headers, json=payload)

        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            return f"DeepSeek API Error: {response.json()}"

    except Exception as e:
        return f"DeepSeek RAG Failed: {str(e)}"
