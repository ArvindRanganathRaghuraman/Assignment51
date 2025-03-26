import json
from langchain.vectorstores import Chroma
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.schema import Document

def index_json_chromadb(json_content, collection_name="json-index", persist_directory="./chroma_langchain_db"):
    """
    Index JSON content (as a string) into ChromaDB.

    Args:
        json_content (str): The JSON content as a string.
        collection_name (str): Name of the ChromaDB collection.
        persist_directory (str): Directory where the ChromaDB database is stored.

    Returns:
        Chroma: The indexed ChromaDB vector store.
    """
    
    # ✅ Initialize ChromaDB
    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"),
        persist_directory=persist_directory
    )

    # ✅ Parse JSON content
    try:
        if isinstance(json_content, str):
            data = json.loads(json_content)  # Convert string to dictionary
        else:
            data = json_content  # Already a dictionary
    except json.JSONDecodeError:
        raise ValueError("❌ Invalid JSON format in the content.")

    # ✅ Extract text chunks safely
    if 'chunks' not in data or not isinstance(data['chunks'], list):
        raise ValueError("❌ No valid chunks found in the JSON content.")
    
    chunks = [chunk if isinstance(chunk, str) else chunk.get("content", "") for chunk in data["chunks"]]
    
    if not chunks:
        raise ValueError("❌ No content found in the JSON chunks.")

    # ✅ Convert chunks to LangChain Documents
    documents = [Document(page_content=chunk, metadata={"source": "in-memory"}) for chunk in chunks if chunk]

    # ✅ Insert documents into ChromaDB
    if documents:
        vector_store.add_documents(documents)
        print(f"✅ Successfully indexed {len(documents)} chunks into ChromaDB ({collection_name}).")
    else:
        print("⚠️ No chunks were indexed. Check if the JSON content contains text.")

    return vector_store
