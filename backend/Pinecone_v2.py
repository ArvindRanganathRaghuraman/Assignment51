import os
import json
from datetime import datetime
from pinecone import Pinecone, ServerlessSpec  
from langchain.vectorstores import Pinecone as PineconeVectorStore
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
from dotenv import load_dotenv  

# Load .env file
load_dotenv(dotenv_path=".env")

def extract_year_quarter_from_filename(filename: str):
    """Extracts Year and Quarter from a filename like '10K10-Q2-2024.json'."""
    parts = filename.split("-")
    if len(parts) >= 3:
        year = parts[-1].split(".")[0]  # Extract year from last part before .json
        quarter = parts[-2]  # Extract quarter (e.g., Q2)
        return year, quarter
    return None, None

def index_json_content(json_content, filename, index_name="json-index", region="us-east-1"):
    """
    Index a JSON content (as string) into Pinecone after chunking, including metadata (Year, Quarter).
    """
    index_name = index_name.lower().replace("_", "-")
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    if not pinecone_api_key:
        raise ValueError("❌ Pinecone API Key is missing! Please check your .env file.")

    # Extract Year and Quarter from filename
    year, quarter = extract_year_quarter_from_filename(filename)

    # Initialize Pinecone Client
    pc = Pinecone(api_key=pinecone_api_key)
    existing_indexes = [index["name"] for index in pc.list_indexes()]

    if index_name not in existing_indexes:
        print(f"⚠️ Index '{index_name}' not found. Creating it now...")
        pc.create_index(
            name=index_name,  
            dimension=384,  # Ensure dimension matches the embedding model
            metric="cosine",  
            spec=ServerlessSpec(cloud="aws", region=region)  
        )
    
    index = pc.Index(index_name)
    embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
    vector_store = PineconeVectorStore(index=index, embedding=embeddings, text_key="page_content")
    
    try:
        data = json.loads(json_content) if isinstance(json_content, str) else json_content
    except json.JSONDecodeError:
        raise ValueError("❌ Invalid JSON format in the content.")

    if 'chunks' not in data or not isinstance(data['chunks'], list):
        raise ValueError("❌ No valid chunks found in the JSON content.")
    
    documents = []
    for chunk in data["chunks"]:
        text = chunk if isinstance(chunk, str) else chunk.get("content", "")
        metadata = {"source": "in-memory"}
        if year and quarter:
            metadata.update({"Year": year, "Quarter": quarter})
        
        if text:
            documents.append(Document(page_content=text, metadata=metadata))
    
    if documents:
        vector_store.add_documents(documents)
        print(f"✅ Successfully indexed {len(documents)} chunks into Pinecone ({index_name}).")
    else:
        print("⚠️ No chunks were created. Check if the JSON content contains text.")
    
    return vector_store
