import os
from pinecone import Pinecone, ServerlessSpec  
from langchain.vectorstores import Pinecone as PineconeVectorStore
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
from Langchain_Chunking import langchain_chunking  # Import chunking function

def index_multiple_mds(md_paths, index_name="nvidia-reports", pinecone_api_key=None, region="us-east-1"):
    """
    Index multiple Markdown (.md) files into Pinecone.

    Args:
        md_paths (list): List of Markdown file paths.
        index_name (str): Name of the Pinecone index.
        pinecone_api_key (str, optional): Pinecone API key (defaults to environment variable).
        region (str, optional): Pinecone region (default: us-east-1).

    Returns:
        PineconeVectorStore: The indexed Pinecone vector store.
    """

    
    if pinecone_api_key is None:
        pinecone_api_key = os.getenv("PINECONE_API_KEY")
    if not pinecone_api_key:
        raise ValueError(" Pinecone API Key is missing! Set it in the environment or pass it as an argument.")

    # ✅ Initialize Pinecone Client
    pc = Pinecone(api_key=pinecone_api_key)

    # ✅ Retrieve list of existing indexes
    existing_indexes = [index["name"] for index in pc.list_indexes()]

    # ✅ If the index does not exist, create it
    if index_name not in existing_indexes:
        print(f"⚠️ Index '{index_name}' not found. Creating it now...")

        pc.create_index(
            name=index_name,  # ✅ FIXED: Explicitly pass the name argument
            dimension=384,  # ✅ FIXED: Ensure the correct number of dimensions (same as embedding model)
            metric="cosine",  # ✅ Set similarity metric
            spec=ServerlessSpec(cloud="aws", region=region)  # ✅ Ensure correct region
        )

    # ✅ Retrieve the correct Pinecone Index object
    index = pc.Index(index_name)

    # ✅ Load Embeddings
    embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)

    # ✅ Initialize the Vector Store
    vector_store = PineconeVectorStore(
        index=index,
        embedding=embeddings,
        text_key="page_content",  # ✅ Specify the text key for the documents
    )

    # ✅ Process each Markdown file and insert into Pinecone
    all_documents = []
    for md_path in md_paths:
        print(f"📄 Processing Markdown File: {md_path}")
        chunks = langchain_chunking(md_path)

        # Convert to LangChain Documents
        documents = [Document(page_content=chunk, metadata={"source": md_path}) for chunk in chunks]
        all_documents.extend(documents)

        print(f"✅ {len(chunks)} chunks created from {md_path}")

    # ✅ Insert all documents into Pinecone
    if all_documents:
        vector_store.add_documents(all_documents)
        print(f"✅ Successfully indexed {len(all_documents)} chunks into Pinecone ({index_name}).")
    else:
        print("⚠️ No chunks were created. Check if Markdown files contain text.")

    return vector_store
