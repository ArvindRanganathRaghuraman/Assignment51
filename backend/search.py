import json
import numpy as np
from numpy.linalg import norm
import openai
from dotenv import load_dotenv
import os

# Load environment variables and configure API
load_dotenv(dotenv_path=".env")
openai.api_key = os.getenv("OPENAI_API_KEY")

def get_embedding(text):
    """Generates an embedding using OpenAI's API."""
    response = openai.embeddings.create(input=[text], model="text-embedding-ada-002")
    return response.data[0].embedding

def cosine_similarity(vec1, vec2):
    """Compute cosine similarity between two vectors."""
    return np.dot(vec1, vec2) / (norm(vec1) * norm(vec2))

def search_from_content(content, query, quarter_filter=None, top_n=5):
    """
    Perform a search on the provided content.
    
    The function filters data by quarter (if specified), generates the query embedding,
    and then calculates similarity scores against each chunk.
    """
    if isinstance(content, dict):
        content = [content]

    # Filter by quarter, if applicable
    filtered_data = [
        item for item in content
        if quarter_filter is None or item.get("quarter") == quarter_filter
    ]
    
    # Generate embedding for the query
    query_embedding = get_embedding(query)

    results = []
    for item in filtered_data:
        # Extract and parse the "text" field
        text_data = item.get("text", "{}")
        try:
            parsed_text = json.loads(text_data)  # Parse JSON string
            chunks = parsed_text.get("chunks", [])  # Extract chunks
        except json.JSONDecodeError:
            print("❌ Failed to decode JSON from text field!")
            chunks = []

        if not chunks:
            print("⚠️ No chunks found for item:", item)
            continue

        # Process each chunk
        for chunk in chunks:
            chunk_embedding = get_embedding(chunk)  # Generate embedding per chunk
            similarity = cosine_similarity(query_embedding, chunk_embedding)
            results.append({"similarity": similarity, "chunk": chunk})

    # Sort by similarity and return the top N results
    results = sorted(results, key=lambda x: x["similarity"], reverse=True)
    print("🔍 Search Results:", results[:top_n])
    return results[:top_n]

def generate_response(query, retrieved_chunks):
    """
    Generates a response using GPT-40-mini with the retrieved chunks as context.
    """
    if not retrieved_chunks:
        return "No relevant information found."

    # Combine retrieved text chunks as context
    context = "\n".join(chunk["chunk"] for chunk in retrieved_chunks)
    prompt = f"""You are an AI assistant. Use the following context to answer the question:

{context}

Question: {query}
Answer:"""

    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content
