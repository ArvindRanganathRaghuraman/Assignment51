import os
from pinecone import Pinecone
import google.generativeai as genai
from dotenv import load_dotenv
from pinecone_indexing import get_huggingface_embedding  # ✅ Reuses existing embedding function

# Load environment variables
load_dotenv(dotenv_path=".env")

# ✅ Configure Google Gemini API
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# ✅ Ensure API key is available
if not GEMINI_API_KEY:
    raise ValueError("Error: GEMINI_API_KEY is missing! Please check your .env file.")

genai.configure(api_key=GEMINI_API_KEY)

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

def extract_text_from_gemini(response):
    """Extracts the text properly from the Gemini API response."""
    if response and hasattr(response, "candidates") and response.candidates:
        candidate = response.candidates[0]
        
        # ✅ Ensure candidate content exists and has parts
        if hasattr(candidate, "content") and hasattr(candidate.content, "parts"):
            extracted_text = []
            for part in candidate.content.parts:
                if hasattr(part, "text"):
                    extracted_text.append(part.text)  # ✅ Properly extract parts
            return " ".join(extracted_text) if extracted_text else "Error: No valid response from Gemini."

    return "Error: Unexpected response format from Gemini API."

def answer_question_gemini(question):
    """Use Google Gemini 2.0 Flash to answer a question based on Markdown content stored in Pinecone."""
    context = retrieve_relevant_chunks(question)
    
    if not context:
        return "No relevant information found in the indexed Markdown document."

    try:
        # ✅ Use Google Gemini API for generating answers
        model = genai.GenerativeModel("gemini-2.0-flash")  # ✅ Updated to Gemini 2.0 Flash
        response = model.generate_content(
            f"Context:\n{context}\n\nAnswer the following question based ONLY on the document above:\n{question}"
        )

        # ✅ Debugging: Print raw response for debugging
        print(f"Raw Gemini Response: {response}")

        return extract_text_from_gemini(response)  # ✅ Extract response properly

    except Exception as e:
        return f"Error: {e}"
