import os
import google.generativeai as genai
from hybrid_search_pinecone import hybrid_search_pinecone
from POCs.hybrid_search_chromadb_and_retrieval import hybrid_search_chroma  # Import Pinecone hybrid search
from dotenv import load_dotenv  # Load environment variables

# ✅ Load API Key
load_dotenv(dotenv_path="/Users/arvindranganathraghuraman/Documents/Assignment4-1/POCs/.env")
gemini_api_key = os.getenv("GEMINI_API_KEY")

def generate_response_gemini_flash(query, quarter_md, top_k=3, index_name="nvidia-reports", file_paths=None, pinecone_api_key=None):
    """
    Uses Google Gemini Flash AI model to answer a query based on retrieved hybrid search results from Pinecone.

    Args:
        query (str): User query.
        quarter_md (str): Name of the specific quarter's Markdown file (e.g., "Q1-2025.md").
        top_k (int): Number of top results to retrieve.
        index_name (str): Pinecone index name.
        file_paths (list, optional): List of Markdown files to index if needed.
        pinecone_api_key (str, optional): Pinecone API key.

    Returns:
        str: Gemini Flash-generated response.
    """

    # ✅ Ensure `file_paths` is provided
    if file_paths is None:
        raise ValueError("❌ `file_paths` cannot be None. Provide a list of Markdown files for indexing.")

    # ✅ Perform Hybrid Search (Retrieve relevant context from Pinecone)
    results = hybrid_search_pinecone(query, quarter_md, file_paths, top_k=top_k, index_name=index_name, pinecone_api_key=pinecone_api_key)

    # ✅ Extract Context from Retrieved Documents
    context = "\n\n".join([doc.page_content for doc in results])

    # ✅ Debugging: Print Retrieved Context
    print("\n🔍 **Retrieved Context from Pinecone:**")
    print(context[:500] if context else "⚠️ No relevant context found!")

    # ✅ Ensure API Key is Loaded
    if not gemini_api_key:
        raise ValueError("❌ GEMINI_API_KEY is missing! Set it in .env or pass it as a parameter.")

    # ✅ If no context is found, return a fallback response
    if not context.strip():
        return "⚠️ No relevant information found in Pinecone. Please check if the index is correctly populated."

    # ✅ Initialize Google Gemini Flash Client
    genai.configure(api_key=gemini_api_key)
    model = genai.GenerativeModel("gemini-1.5-flash")

    # ✅ Call Gemini Flash AI
    try:
        response = model.generate_content(f"Use the following context to answer the question:\n\nContext:\n{context}\n\nQuestion: {query}")

        return response.text

    except Exception as e:
        print(f"❌ Gemini API Error: {e}")
        return "⚠️ There was an issue with generating a response from Gemini Flash."


# ✅ Define the test parameters
test_query = "What was NVIDIA's revenue in Q1 2023?"
test_quarter_md = "POCs/DOCLING_PDF_PLUMBER_Markdowns/10K10Q-Q1-2023-with-image-refs.md"
test_index_name = "nvidia-reports"

# ✅ Define the Markdown file paths for indexing
test_file_paths = [
    "POCs/DOCLING_PDF_PLUMBER_Markdowns/10K10Q-Q1-2023-with-image-refs.md",
    "POCs/DOCLING_PDF_PLUMBER_Markdowns/10K10Q-Q2-2023-with-image-refs.md",
    "POCs/DOCLING_PDF_PLUMBER_Markdowns/10K10Q-Q3-2023-with-image-refs.md",
    "POCs/DOCLING_PDF_PLUMBER_Markdowns/10K10Q-Q4-2023-with-image-refs.md"
]

# ✅ Run the function
print("\n🚀 Running Test: Gemini Flash Response Generation with Pinecone Hybrid Search...")
response_gemini_flash = generate_response_gemini_flash(test_query, test_quarter_md, top_k=3, index_name=test_index_name, file_paths=test_file_paths)


