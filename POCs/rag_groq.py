import os
from openai import OpenAI
from dotenv import load_dotenv
from pinecone import Pinecone
from pinecone_indexing import get_huggingface_embedding  # ✅ Reuses existing embedding function

# Load environment variables
load_dotenv(dotenv_path=".env")

# ✅ Set up Groq API client
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

client = OpenAI(
    api_key=GROQ_API_KEY,
    base_url="https://api.x.ai/v1",  # ✅ Groq API OpenAI-compatible base URL
)

# ✅ Initialize Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

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

    # ✅ Handle case where no relevant documents are found
    if not search_results.get("matches"):
        return ""

    retrieved_texts = [match["metadata"]["text"] for match in search_results["matches"]]
    return "\n".join(retrieved_texts)

def answer_question_groq(question):
    """Use Groq API via OpenAI SDK to answer a question based on Markdown content stored in Pinecone."""
    context = retrieve_relevant_chunks(question)
    
    if not context:
        return "No relevant information found in the indexed Markdown document."

    try:
        completion = client.chat.completions.create(
            model="grok-2",  # ✅ Using Groq-supported Mixtral model
            messages=[
                {"role": "system", "content": "You are an AI assistant that answers questions based on the given document."},
                {"role": "user", "content": f"Context:\n{context}\n\nAnswer the following question based ONLY on the document above:\n{question}"}
            ],
            temperature=0.7
        )

        return completion.choices[0].message.content  # ✅ Extract and return response

    except Exception as e:
        return f"Error: {e}"
