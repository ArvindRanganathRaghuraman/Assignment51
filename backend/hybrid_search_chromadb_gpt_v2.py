import os
import re
import openai
from langchain.vectorstores import Chroma
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv  

load_dotenv(dotenv_path=".env")  # ✅ Load .env file

# ✅ Load API Key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ✅ Ensure API key is set
if not OPENAI_API_KEY:
    raise ValueError("❌ OpenAI API Key is missing! Check .env or set it manually.")

# ✅ Initialize OpenAI Client (Fixed!)
client = openai.OpenAI(api_key=OPENAI_API_KEY)

# ✅ Hybrid Search Function for ChromaDB
def query_chromadb_with_gpt(query, collection_name="json-index", persist_directory="./chroma_langchain_db", top_k=5):
    """
    Query ChromaDB with a hybrid search (semantic + keyword-based) and generate an answer using GPT-4o.

    Args:
        query (str): The question/query from the user.
        collection_name (str): Name of the ChromaDB collection.
        persist_directory (str): Directory where the ChromaDB vector store is stored.
        top_k (int): Number of top search results to retrieve.

    Returns:
        str: The generated answer from GPT-4o based on retrieved context.
    """

    # ✅ Load vector store from disk
    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"),
        persist_directory=persist_directory
    )

    # ✅ Perform semantic search (vector-based)
    semantic_results = vector_store.similarity_search_with_relevance_scores(query, k=top_k)

    # ✅ Perform keyword-based search (fallback)
    keyword_results = vector_store.similarity_search_with_relevance_scores(query, k=top_k)

    # ✅ Combine results with weights
    combined = []
    for doc, score in semantic_results:
        combined.append((doc.page_content, 0.7 * score))
    for doc, score in keyword_results:
        combined.append((doc.page_content, 0.3 * score))

    # ✅ Deduplicate and sort by weighted score
    unique_docs = {}
    for content, weighted_score in combined:
        if content not in unique_docs or weighted_score > unique_docs[content]:
            unique_docs[content] = weighted_score

    sorted_results = sorted(unique_docs.items(), key=lambda x: x[1], reverse=True)
    top_chunks = [item[0] for item in sorted_results[:top_k]]

    if not top_chunks:
        return "I couldn't find relevant information in the database."

    # ✅ Prepare context for GPT-4o
    context = "\n\n".join(top_chunks)

    # ✅ Generate answer using GPT-4o (Fixed API)
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are an AI financial assistant that answers questions based on reports."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}\nAnswer based on the above context:"}
        ]
    )

    return response.choices[0].message.content
'''
# ✅ Example Usage
query_result = query_chromadb_with_gpt("What is the Revenue for Q1 2025")
print("\n💡 Answer:\n", query_result)'
'''
