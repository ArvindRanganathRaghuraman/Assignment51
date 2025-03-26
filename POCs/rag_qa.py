import os
from pinecone import Pinecone  # ✅ Correct import
from dotenv import load_dotenv
from litellm import completion
from pinecone_indexing import get_huggingface_embedding  # ✅ Reuses existing embedding function

# Load environment variables
load_dotenv(dotenv_path=".env")

# Initialize Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))  # ✅ Correct way to initialize

# Define index name
INDEX_NAME = os.getenv("PINECONE_INDEX")

# Ensure the index exists
if INDEX_NAME not in pc.list_indexes().names():
    raise ValueError(f"Error: The Pinecone index '{INDEX_NAME}' does not exist. Please create it first.")

# Connect to the existing index
index = pc.Index(INDEX_NAME)

def retrieve_relevant_chunks(question, top_k=5):
    """Retrieve the most relevant chunks from Pinecone based on the Markdown content."""
    query_embedding = get_huggingface_embedding(question)  # ✅ Use the same embedding function

    search_results = index.query(
        vector=query_embedding, 
        top_k=top_k, 
        include_metadata=True
    )

    retrieved_texts = [match["metadata"]["text"] for match in search_results["matches"]]
    return "\n".join(retrieved_texts)

def answer_question_gpt(question):
    """Use GPT-4o to answer a question based on Markdown content stored in Pinecone."""
    context = retrieve_relevant_chunks(question)
    
    if not context:
        return "No relevant information found in the indexed Markdown document."

    response = completion(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are an AI assistant that answers questions based on the given document."},
            {"role": "user", "content": f"Context:\n{context}\n\nAnswer the following question based ONLY on the document above:\n{question}"}
        ]
    )

    return response['choices'][0]['message']['content']
