import os
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from Pinecone_POCS import index_multiple_mds  # Import the function to index multiple files

def hybrid_search_pinecone(query, quarter_file, file_paths, top_k=3, index_name="nvidia-reports", pinecone_api_key=None):
    """
    TRUE Hybrid search: 
    - First checks if Pinecone index exists, if not, it creates it and indexes files.
    - Then performs vector similarity retrieval on the filtered results.

    Args:
        query (str): User query.
        quarter_file (str): Name of the specific quarter's file to filter (e.g., "Q1-2025.md").
        file_paths (list): List of file paths (.pdf or .md) to index if the index does not exist.
        top_k (int): Number of top results to return.
        index_name (str): Name of the Pinecone index.
        pinecone_api_key (str, optional): Pinecone API key.

    Returns:
        list[Document]: Retrieved documents ranked by relevance.
    """

    # ✅ Load API Key
    if pinecone_api_key is None:
        pinecone_api_key = os.getenv("PINECONE_API_KEY")
    if not pinecone_api_key:
        raise ValueError("❌ Pinecone API Key is missing! Set it in the environment or pass it as an argument.")

    # ✅ Initialize Pinecone Client
    pc = Pinecone(api_key=pinecone_api_key)

    # ✅ Check if index exists, otherwise create and populate it
    existing_indexes = pc.list_indexes()
    if index_name not in existing_indexes:
        print(f"⚠️ Index '{index_name}' not found. Creating and indexing files first...")
        vector_store = index_multiple_mds(file_paths, index_name, pinecone_api_key)
    else:
        print(f"✅ Using existing Pinecone index: {index_name}")

    # ✅ Retrieve the correct Pinecone Index object
    index = pc.Index(index_name)  # ✅ FIXED: Get the actual Index object

    # ✅ Load Embeddings Separately
    embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)

    # ✅ Initialize the Vector Store with the Correct Index Object
    vector_store = PineconeVectorStore(index=index, embedding=embeddings, text_key="page_content")  # ✅ FIXED

    # ✅ TRUE Hybrid Search
    # Step 1: **Filter by metadata (quarterly report)**
    filters = {"source": {"$eq": quarter_file}}  # Only search within the selected quarter
    
    # Step 2: **Perform vector similarity search within the filtered results**
    results = vector_store.similarity_search(query, k=top_k, filter=filters)

    # ✅ Display Results
    print(f"\n🔍 Hybrid Search Results from {quarter_file} (Pinecone):")
    for doc in results:
        print(doc.page_content[:500])  # Print first 500 characters for preview

    return results
