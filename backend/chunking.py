import os
import json
import nltk
import tiktoken  # Import OpenAI's tokenizer for better chunking
from nltk.tokenize import sent_tokenize
from io import BytesIO
from gcs_utils import upload_to_gcs  # Import the upload function
from langchain.text_splitter import RecursiveCharacterTextSplitter

nltk.download('punkt')

# Initialize tokenizer for accurate token count
tokenizer = tiktoken.encoding_for_model("text-embedding-ada-002")
MAX_TOKENS = 8192  # Maximum token limit for OpenAI embeddings

def count_tokens(text):
    """Returns the number of tokens in a given text."""
    return len(tokenizer.encode(text))

# LangChain Chunking Function (Modified to handle raw text)
def langchain_chunking(text, chunk_size=512, chunk_overlap=50):
    """Uses LangChain's RecursiveCharacterTextSplitter to chunk the text."""
    if not text:
        raise ValueError("Text input cannot be empty.")
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.split_text(text)

# 1. Fixed-size chunking (between 200-400 tokens)
def chunk_fixed_size(text, chunk_size=300):
    """Splits text into fixed-size token chunks (not word-based)."""
    tokens = tokenizer.encode(text)
    return [tokenizer.decode(tokens[i:i + chunk_size]) for i in range(0, len(tokens), chunk_size)]

# 2. Semantic-based chunking (split at sentence boundaries)
def chunk_by_sentences(text, max_tokens=400):
    """Chunks text by sentence while ensuring token limit per chunk."""
    sentences = sent_tokenize(text)
    chunks, current_chunk = [], ""

    for sentence in sentences:
        if count_tokens(current_chunk + sentence) <= max_tokens:
            current_chunk += " " + sentence
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence  # Start new chunk

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks

# 3. Sliding window chunking (overlapping chunks)
def chunk_sliding_window(text, chunk_size=300, overlap=50):
    """Creates overlapping chunks using tokens (prevents loss of data)."""
    tokens = tokenizer.encode(text)
    step = chunk_size - overlap  # Ensure overlap consistency
    return [tokenizer.decode(tokens[i: i + chunk_size]) for i in range(0, len(tokens), step) if i < len(tokens)]

# 4. Recursive chunking (hierarchical splitting with token validation)
def chunk_recursive(text, chunk_size=300, overlap=50, separators=["\n\n", ".", "?", "!", "\n", " "]):
    """
    Recursively chunks text based on logical separators while ensuring minimal overlap.
    Adjusted for structured text like reports & long-form documents.
    """
    if count_tokens(text) <= chunk_size:
        return [text]

    for separator in separators:
        parts = text.split(separator)
        chunks, temp_chunk = [], ""

        for part in parts:
            if count_tokens(temp_chunk + part) <= chunk_size:
                temp_chunk += part + separator
            else:
                chunks.append(temp_chunk.strip())
                temp_chunk = part + separator

        if temp_chunk:
            chunks.append(temp_chunk.strip())

        # If all chunks are valid, return them
        if all(count_tokens(chunk) <= chunk_size for chunk in chunks):
            return chunks

    # Fallback: Fixed-size chunking if no valid split is found
    print("⚠️ Using fixed-size chunking as fallback.")
    return chunk_fixed_size(text, chunk_size)

# ✅ New Function: Ensure all chunks are within the token limit
def validate_and_split_chunks(chunks, max_tokens=MAX_TOKENS):
    """Ensures no chunk exceeds the max token limit by splitting further if needed."""
    valid_chunks = []

    for chunk in chunks:
        if count_tokens(chunk) > max_tokens:
            print(f"⚠️ Chunk too large ({count_tokens(chunk)} tokens), splitting further...")
            valid_chunks.extend(chunk_recursive(chunk, chunk_size=300))  # Recursive splitting
        else:
            valid_chunks.append(chunk)

    return valid_chunks

def process_and_upload_chunked_data(text, destination_blob_name, strategy="fixed", chunk_size=512, chunk_overlap=50):
    """Process text with the selected chunking strategy and upload chunked data to GCS."""
    
    # Mapping of strategies to the corresponding functions
    chunking_strategies = {
        "fixed": chunk_fixed_size,
        "sentence": chunk_by_sentences,
        "sliding": chunk_sliding_window,
        "recursive": chunk_recursive,
        "langchain": langchain_chunking  # Add LangChain option
    }

    if strategy not in chunking_strategies:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    # Chunk the data using the selected strategy
    if strategy == "langchain":
        # LangChain strategy processes raw text
        chunked_data = langchain_chunking(text, chunk_size, chunk_overlap)
    else:
        # Other strategies process text directly
        chunked_data = chunking_strategies[strategy](text)
    
    # ✅ Validate and split oversized chunks
    chunked_data = validate_and_split_chunks(chunked_data)

    # Prepare output as JSON structure
    chunked_json = {"chunks": chunked_data}
    
    # Prepare the file name and convert chunked data to JSON string
    destination_blob_name = os.path.splitext(destination_blob_name)[0] + ".json"
    chunked_json_str = json.dumps(chunked_json, ensure_ascii=False, indent=4)
    
    # Convert the string to bytes for uploading
    file_stream = BytesIO(chunked_json_str.encode('utf-8'))  # Encode the string to bytes
    
    # Upload the chunked data to GCS
    try:
        file_url = upload_to_gcs(file_stream, destination_blob_name, content_type="application/json")
        print(f"✅ File uploaded successfully to GCS: {file_url}")
        return file_url
    except Exception as e:
        print(f"❌ Error uploading to GCS: {e}")
        raise ValueError(f"Error uploading to GCS: {e}")
