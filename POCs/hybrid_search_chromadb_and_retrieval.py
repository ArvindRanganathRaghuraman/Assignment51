import os
import google.generativeai as genai
from langchain.vectorstores import Chroma
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from openai import OpenAI
from Chromadb_POC import index_multiple_mds_chroma  # Import indexing function
from dotenv import load_dotenv  # Load environment variables

# ✅ Load API Key
load_dotenv(dotenv_path="/Users/arvindranganathraghuraman/Documents/Assignment4-1/POCs/.env")
gemini_api_key = os.getenv("GEMINI_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")

def hybrid_search_chroma(query, quarter_md, top_k=3, collection_name="nvidia-reports", persist_directory="./chroma_langchain_db", md_paths=None):
    """
    TRUE Hybrid search: 
    - If ChromaDB index is missing, first indexes the Markdown files.
    - Then filters ChromaDB results by quarter metadata.
    - Finally, performs vector similarity retrieval on the filtered results.

    Args:
        query (str): User query.
        quarter_md (str): Name of the specific quarter's Markdown file (e.g., "10K10Q-Q1-2025.md").
        top_k (int): Number of top results to return.
        collection_name (str): Name of the ChromaDB collection.
        persist_directory (str): Directory where ChromaDB is stored.
        md_paths (list, optional): List of Markdown file paths to index if needed.

    Returns:
        list[Document]: Retrieved documents ranked by relevance.
    """

    # ✅ Initialize ChromaDB Vector Store
    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"),
        persist_directory=persist_directory
    )

    # ✅ Check if ChromaDB contains any data
    try:
        test_results = vector_store.similarity_search("test", k=1)
        if test_results:
            print(f"✅ Using existing ChromaDB index: {collection_name}")
        else:
            print(f"⚠️ No existing index found in ChromaDB ({collection_name}). Indexing Markdown files first...")
            if md_paths is None:
                raise ValueError("❌ `md_paths` cannot be None. Provide a list of Markdown files for indexing.")
            vector_store = index_multiple_mds_chroma(md_paths, collection_name, persist_directory)

    except Exception as e:
        print(f"⚠️ Error accessing ChromaDB: {e}. Reindexing from Markdown files...")
        if md_paths is None:
            raise ValueError("❌ `md_paths` cannot be None. Provide a list of Markdown files for indexing.")
        vector_store = index_multiple_mds_chroma(md_paths, collection_name, persist_directory)

    # ✅ TRUE Hybrid Search
    # Step 1: **Filter by metadata (quarterly report)**
    filters = {"source": quarter_md}  # Only search within the selected quarter

    # Step 2: **Perform vector similarity search within the filtered results**
    results = vector_store.similarity_search(query, k=top_k, filter=filters)

    # ✅ Display Results
    print(f"\n🔍 Hybrid Search Results from {quarter_md} (ChromaDB):")
    for doc in results:
        print(doc.page_content[:500])  # Print first 500 characters for preview

    return results


def generate_response_gemini_flash(query, quarter_md, top_k=3, collection_name="nvidia-reports", persist_directory="./chroma_langchain_db", md_paths=None):
    """
    Uses Google Gemini Flash AI model to answer a query based on retrieved hybrid search results from ChromaDB.

    Args:
        query (str): User query.
        quarter_md (str): Name of the specific quarter's Markdown file (e.g., "10K10Q-Q1-2025.md").
        top_k (int): Number of top results to retrieve.
        collection_name (str): ChromaDB collection name.
        persist_directory (str): Directory where ChromaDB is stored.
        md_paths (list, optional): List of Markdown files to index if needed.

    Returns:
        str: Gemini Flash-generated response.
    """

    # ✅ Perform Hybrid Search (Retrieve relevant context from ChromaDB)
    results = hybrid_search_chroma(
        query=query,  
        quarter_md=quarter_md,  
        top_k=top_k,  
        collection_name=collection_name,  
        persist_directory=persist_directory,
        md_paths=md_paths  
    )

    # ✅ Extract Context from Retrieved Documents
    context = "\n\n".join([doc.page_content for doc in results])

    # ✅ Debugging: Print Retrieved Context
    print("\n🔍 **Retrieved Context from ChromaDB:**")
    print(context[:500] if context else "⚠️ No relevant context found!")

    # ✅ If no context is found, return a fallback response
    if not context.strip():
        return "⚠️ No relevant information found in ChromaDB. Please check if the index is correctly populated."

    # ✅ Ensure API Key is Loaded
    if not gemini_api_key:
        raise ValueError("❌ GEMINI_API_KEY is missing! Set it in .env or pass it as a parameter.")

    # ✅ Initialize Google Gemini Flash Client
    genai.configure(api_key=gemini_api_key)
    model = genai.GenerativeModel("gemini-1.5-flash")

    # ✅ Call Gemini Flash AI
    try:
        response = model.generate_content(f"Use the following context to answer the question:\n\nContext:\n{context}\n\nQuestion:\n{query}")

        return response.text

    except Exception as e:
        print(f"❌ Gemini API Error: {e}")
        return "⚠️ There was an issue with generating a response from Gemini Flash."
    
def generate_response_gpt(query, quarter_md, top_k=3, collection_name="nvidia-reports", persist_directory="./chroma_langchain_db", md_paths=None):
    """
    Uses OpenAI GPT-4 model to answer a query based on retrieved hybrid search results from ChromaDB.

    Args:
        query (str): User query.
        quarter_md (str): Name of the specific quarter's Markdown file (e.g., "10K10Q-Q1-2025.md").
        top_k (int): Number of top results to retrieve.
        collection_name (str): ChromaDB collection name.
        persist_directory (str): Directory where ChromaDB is stored.
        md_paths (list, optional): List of Markdown files to index if needed.

    Returns:
        str: GPT-4 generated response.
    """

    # ✅ Perform Hybrid Search (Retrieve relevant context from ChromaDB)
    results = hybrid_search_chroma(
        query=query,  
        quarter_md=quarter_md,  
        top_k=top_k,  
        collection_name=collection_name,  
        persist_directory=persist_directory,
        md_paths=md_paths  
    )

    # ✅ Extract Context from Retrieved Documents
    context = "\n\n".join([doc.page_content for doc in results])

    # ✅ Debugging: Print Retrieved Context
    print("\n🔍 **Retrieved Context from ChromaDB:**")
    print(context[:500] if context else "⚠️ No relevant context found!")

    # ✅ If no context is found, return a fallback response
    if not context.strip():
        return "⚠️ No relevant information found in ChromaDB. Please check if the index is correctly populated."

    # ✅ Ensure API Key is Loaded
    if not openai_api_key:
        raise ValueError("❌ OPENAI_API_KEY is missing! Set it in .env or pass it as a parameter.")

    # ✅ Initialize OpenAI GPT Client
    client = OpenAI(api_key=openai_api_key)

    # ✅ Call OpenAI GPT-4 API
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an AI specializing in financial analysis. Use the retrieved context to generate an accurate answer."},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion:\n{query}"}
            ],
            temperature=0.0
        )

        return response.choices[0].message.content

    except Exception as e:
        print(f"❌ OpenAI API Error: {e}")
        return "⚠️ There was an issue with generating a response from GPT-4."


# ✅ Define the test parameters
'''test_query = "What was NVIDIA's revenue in Q1 2025?"
test_quarter_md = "POCs/DOCLING_PDF_PLUMBER_Markdowns/10K10Q-Q1-2025-with-image-refs.md"
test_collection_name = "nvidia-reports"
test_persist_directory = "./chroma_langchain_db"

# ✅ Define the Markdown file paths for indexing
test_md_paths = [
    "POCs/DOCLING_PDF_PLUMBER_Markdowns/10K10Q-Q1-2025-with-image-refs.md",
    "POCs/DOCLING_PDF_PLUMBER_Markdowns/10K10Q-Q2-2025-with-image-refs.md",
    "POCs/DOCLING_PDF_PLUMBER_Markdowns/10K10Q-Q3-2025-with-image-refs.md",
    "POCs/DOCLING_PDF_PLUMBER_Markdowns/10K10Q-Q4-2025-with-image-refs.md"
]

# ✅ Run the function
print("\n🚀 Running Test: Gemini Flash Response Generation with ChromaDB Hybrid Search...")
response_gemini = generate_response_gemini_flash(
    query=test_query, 
    quarter_md=test_quarter_md, 
    top_k=3, 
    collection_name=test_collection_name, 
    persist_directory=test_persist_directory,
    md_paths=test_md_paths
)

# ✅ Print the output
print("\n💬 **Gemini Flash Response (ChromaDB):**")
print(response_gemini)'''



test_query = "What was NVIDIA's revenue in Q1 2025?"
test_quarter_md = "POCs/DOCLING_PDF_PLUMBER_Markdowns/10K10Q-Q1-2025-with-image-refs.md"
test_collection_name = "nvidia-reports"
test_persist_directory = "./chroma_langchain_db"

# ✅ Define the Markdown file paths for indexing
test_md_paths = [
    "POCs/DOCLING_PDF_PLUMBER_Markdowns/10K10Q-Q1-2025-with-image-refs.md",
    "POCs/DOCLING_PDF_PLUMBER_Markdowns/10K10Q-Q2-2025-with-image-refs.md",
    "POCs/DOCLING_PDF_PLUMBER_Markdowns/10K10Q-Q3-2025-with-image-refs.md",
    "POCs/DOCLING_PDF_PLUMBER_Markdowns/10K10Q-Q4-2025-with-image-refs.md"
]

# ✅ Run the function
print("\n🚀 Running Test: GPT-4 Response Generation with ChromaDB Hybrid Search...")
response_gpt = generate_response_gpt(
    query=test_query, 
    quarter_md=test_quarter_md, 
    top_k=3, 
    collection_name=test_collection_name, 
    persist_directory=test_persist_directory,
    md_paths=test_md_paths
)

# ✅ Print the output
print("\n💬 **GPT-4 Response (ChromaDB):**")
print(response_gpt)
