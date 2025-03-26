import os
from langchain.vectorstores import Chroma
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
from Langchain_Chunking import langchain_chunking  # Import chunking function

def index_multiple_mds_chroma(md_paths, collection_name="nvidia-reports", persist_directory="./chroma_langchain_db"):
    """
    Index multiple Markdown (.md) files into ChromaDB.

    Args:
        md_paths (list): List of Markdown file paths.
        collection_name (str): Name of the ChromaDB collection.
        persist_directory (str): Directory where the ChromaDB database is stored.

    Returns:
        Chroma: The indexed ChromaDB vector store.
    """

    # ✅ Validate file extensions (Only .md files allowed)
    for md_path in md_paths:
        if not md_path.endswith(".md"):
            raise ValueError(f"❌ Invalid file format: {md_path}. Only Markdown (.md) files are allowed.")

    # ✅ Initialize ChromaDB
    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"),
        persist_directory=persist_directory
    )

    # ✅ Process each Markdown file and insert into ChromaDB
    all_documents = []
    for md_path in md_paths:
        print(f"📄 Processing Markdown File: {md_path}")
        chunks = langchain_chunking(md_path)

        # Convert to LangChain Documents
        documents = [Document(page_content=chunk, metadata={"source": md_path}) for chunk in chunks]
        all_documents.extend(documents)

        print(f"✅ {len(chunks)} chunks created from {md_path}")

    # ✅ Insert all documents into ChromaDB
    if all_documents:
        vector_store.add_documents(all_documents)
        print(f"✅ Successfully indexed {len(all_documents)} chunks into ChromaDB ({collection_name}).")
    else:
        print("⚠️ No chunks were created. Check if Markdown files contain text.")

    return vector_store
