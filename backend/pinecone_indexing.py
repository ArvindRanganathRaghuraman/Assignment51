import os
import torch
from transformers import AutoModel, AutoTokenizer
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv

# Load environment variables
load_dotenv(dotenv_path=".env")

# Initialize Pinecone with the new method
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))



# Define index name
INDEX_NAME = os.getenv("PINECONE_INDEX")

# Ensure the index exists
if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME, 
        dimension=768,  # Adjust based on embedding model
        metric='cosine',
        spec=ServerlessSpec(
            cloud='aws',
            region='us-east-1'
        )
    )

# Connect to the existing index
index = pc.Index(INDEX_NAME)

# ✅ Load a lightweight Hugging Face embedding model
MODEL_NAME = "sentence-transformers/all-distilroberta-v1"  # ✅ Small and fast embedding model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)

def get_huggingface_embedding(text):
    """Generate embeddings using a lightweight Hugging Face model."""
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        output = model(**inputs)
        return output.last_hidden_state[:, 0, :].squeeze().tolist()  # Extract sentence embedding

def split_text(text, chunk_size=500, overlap=100):
    """Manually split text into smaller chunks for indexing."""
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunks.append(text[i:i + chunk_size])
    return chunks

def index_markdown_data(markdown_text, file_name):
    """Indexes Markdown content in Pinecone using a lightweight embedding model."""
    chunks = split_text(markdown_text)

    # Generate embeddings and store in Pinecone
    vectors = []
    for i, chunk in enumerate(chunks):
        embedding = get_huggingface_embedding(chunk)
        vectors.append((f"{file_name}_chunk_{i}", embedding, {"text": chunk}))

    index.upsert(vectors)  # ✅ Upload all embeddings at once

    print(f"Indexed {len(chunks)} chunks from {file_name} into Pinecone.")
