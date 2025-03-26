import openai
import json
import re
import os
from dotenv import load_dotenv
from io import BytesIO
from gcs_utils import upload_to_gcs  # Import the GCS upload function

# Load environment variables from .env file
load_dotenv(dotenv_path=".env")

# Set OpenAI API key globally using the environment variable
openai.api_key = os.getenv('OPENAI_API_KEY')

def get_embedding(text):
    """Generates embedding using OpenAI model with the new SDK syntax."""
    response = openai.embeddings.create(
        input=[text],  
        model="text-embedding-ada-002"
    )
    return response.data[0].embedding

def process_and_store_embeddings(content_dict, original_file_name):
    """Generates embeddings, stores them in memory, and uploads to GCS with a cleaned-up file name."""
    
    all_chunks = []
    
    for filename, chunks in content_dict.items():
        quarter = re.search(r'Q\d_\d{4}', filename)  # Extract quarter info from filename
        quarter = quarter.group() if quarter else "Unknown"

        for chunk in chunks:
            embedding = get_embedding(chunk)
            all_chunks.append({
                "text": chunk,
                "embedding": embedding,
                "filename": filename,
                "quarter": quarter
            })

    # Convert processed embeddings to JSON
    embeddings_json_str = json.dumps(all_chunks, ensure_ascii=False, indent=4)
    
    # Convert string to bytes for uploading
    file_stream = BytesIO(embeddings_json_str.encode('utf-8'))  

    # ✅ Extract only the base name (remove .pdf.json or .json)
    base_name = os.path.basename(original_file_name)  # Extract file name
    base_name = re.sub(r"\.pdf\.json$|\.json$", "", base_name)  # Remove `.pdf.json` or `.json`
    cleaned_file_name = f"{base_name}.json"  # Add only `.json`

    destination_blob_name = f"embeddings/{cleaned_file_name}"  # Store under /embeddings/

    # Upload the embeddings file to GCS
    try:
        file_url = upload_to_gcs(file_stream, destination_blob_name, content_type="application/json")
        print(f"✅ Embeddings uploaded successfully to GCS: {file_url}")
        return file_url
    except Exception as e:
        print(f"❌ Error uploading to GCS: {e}")
        raise ValueError(f"Error uploading to GCS: {e}")

if __name__ == "__main__":
    pass  # This script will be called from main.py
