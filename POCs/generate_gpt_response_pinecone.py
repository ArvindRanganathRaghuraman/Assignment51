import os
import openai
from openai import OpenAI
from hybrid_search_pinecone import hybrid_search_pinecone
from POCs.hybrid_search_chromadb_and_retrieval  import hybrid_search_chroma # Import Pinecone hybrid search
from dotenv import load_dotenv  # Load environment variables

# ✅ Load API Key
load_dotenv(dotenv_path="/Users/arvindranganathraghuraman/Documents/Assignment4-1/POCs/.env")
openai_api_key = os.getenv("OPENAI_API_KEY")

def generate_response_gpt_pinecone(query, quarter_md, top_k=3, index_name="nvidia-reports", file_paths=None, pinecone_api_key=None):
    """
    Uses OpenAI GPT model to answer a query based on retrieved hybrid search results from Pinecone.

    Args:
        query (str): User query.
        quarter_md (str): Name of the specific quarter's Markdown file (e.g., "Q1-2025.md").
        top_k (int): Number of top results to retrieve.
        index_name (str): Pinecone index name.
        file_paths (list, optional): List of Markdown files to index if needed.
        pinecone_api_key (str, optional): Pinecone API key.

    Returns:
        str: GPT-4 generated response.
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
    if not openai_api_key:
        raise ValueError("❌ OPENAI_API_KEY is missing! Set it in .env or pass it as a parameter.")

    # ✅ If no context is found, return a fallback response
    if not context.strip():
        return "⚠️ No relevant information found in Pinecone. Please check if the index is correctly populated."
    
    client = OpenAI(api_key=openai_api_key)

    # ✅ Call OpenAI's GPT-4 API
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an AI specializing in financial analysis. Use the retrieved context to generate an accurate answer."},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
            ],
            temperature=0.0
        )

        return response["choices"][0]["message"]["content"]

    except Exception as e:
        print(f"❌ OpenAI API Error: {e}")
        return "⚠️ There was an issue with generating a response from GPT-4."

# ✅ Define the test parameters
test_query = "What was NVIDIA's revenue in Q1 2025?"
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
print("\n🚀 Running Test: GPT-4 Response Generation with Pinecone Hybrid Search...")
response_gpt = generate_response_gpt_pinecone(test_query, test_quarter_md, top_k=3, index_name=test_index_name, file_paths=test_file_paths)

# ✅ Print the output
print("\n💬 **GPT-4 Response:**")
print(response_gpt)


def generate_response_gpt_chroma(query, quarter_md, top_k=3, collection_name="nvidia-reports", persist_directory="./chroma_langchain_db"):
    """
    Uses OpenAI GPT model to answer a query based on retrieved hybrid search results from ChromaDB.

    Args:
        query (str): User query.
        quarter_md (str): Name of the specific quarter's Markdown file (e.g., "Q1-2025.md").
        top_k (int): Number of top results to retrieve.
        collection_name (str): ChromaDB collection name.
        persist_directory (str): Directory where ChromaDB is stored.

    Returns:
        str: GPT-4 generated response.
    """

    # ✅ Perform Hybrid Search (Retrieve relevant context from ChromaDB)
    results = hybrid_search_chroma(
        query=query, 
        quarter_md=quarter_md, 
        top_k=top_k, 
        collection_name=collection_name, 
        persist_directory=persist_directory
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
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
            ],
            temperature=0.0
        )

        return response.choices[0].message.content

    except Exception as e:
        print(f"❌ OpenAI API Error: {e}")
        return "⚠️ There was an issue with generating a response from GPT-4."
    

# ✅ Define the test parameters
test_query = "What was NVIDIA's revenue in Q1 2023?"
test_quarter_md = "POCs/DOCLING_PDF_PLUMBER_Markdowns/10K10Q-Q1-2023-with-image-refs.md"
test_collection_name = "nvidia-reports"

# ✅ Run the function
print("\n🚀 Running Test: GPT-4 Response Generation with ChromaDB Hybrid Search...")
response_gpt_chroma = generate_response_gpt_chroma(
    query=test_query, 
    quarter_md=test_quarter_md, 
    top_k=3, 
    collection_name=test_collection_name
)

# ✅ Print the output
print("\n💬 **GPT-4 Response (ChromaDB):**")
print(response_gpt_chroma)


