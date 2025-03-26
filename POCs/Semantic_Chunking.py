import PyPDF2
from sentence_transformers import SentenceTransformer
import numpy as np

# Function to read PDF and return text
def read_pdf(pdf_path):
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        text = ""
        for page in range(len(reader.pages)):
            text += reader.pages[page].extract_text()
    return text

# Function to chunk text semantically
def semantic_chunking(pdf_path, chunk_size=512):
    text = read_pdf(pdf_path)
    
    # Initialize SentenceTransformer for semantic embeddings
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Tokenize text into sentences
    sentences = text.split(". ")  # Simple sentence splitting
    
    # Generate embeddings for each sentence
    embeddings = model.encode(sentences)
    
    # Create chunks based on the size of embeddings
    chunks = []
    current_chunk = []
    current_embedding = []
    
    for sentence, embedding in zip(sentences, embeddings):
        if len(current_chunk) < chunk_size:
            current_chunk.append(sentence)
            current_embedding.append(embedding)
        else:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]
            current_embedding = [embedding]
    
    if current_chunk:
        chunks.append(" ".join(current_chunk))  # Add the last chunk
    
    return chunks

# Example usage:
pdf_path = "10K10Q-Quarter-1-2025.pdf"
chunks = semantic_chunking(pdf_path)

for chunk in chunks:
    print(chunk)
