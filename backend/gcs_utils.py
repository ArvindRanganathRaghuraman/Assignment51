from google.cloud import storage
from fastapi.responses import StreamingResponse
import io

# Set your GCS bucket name
BUCKET_NAME = "pdfstorage_1"

# Initialize the GCS client
storage_client = storage.Client()
bucket = storage_client.bucket(BUCKET_NAME)

def upload_to_gcs(file_stream, destination_blob_name: str, content_type: str = "text/markdown") -> str:
    """Uploads an in-memory file to Google Cloud Storage and returns the file URL."""
    
    # Ensure the file is saved directly under the `outputs/` folder
    destination_blob_name = f"{destination_blob_name}"  # No need for pdf_files/ folder
    
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_file(file_stream, content_type=content_type)
    
    return f"https://storage.googleapis.com/{BUCKET_NAME}/{destination_blob_name}"

def list_files_in_gcs(folder_name: str = ""):
    """Lists all files in the specified folder in the GCS bucket."""
    
    # Filter by folder prefix
    prefix = f"{folder_name}/" if folder_name else ""
    
    files = [blob.name for blob in bucket.list_blobs(prefix=prefix)]
    
    return files

def get_file_content(file_name):
    """Fetches the content of a markdown file from GCS."""
    blob = bucket.blob(file_name)
    return blob.download_as_text()

def download_file_from_gcs(file_name):
    """Fetches the file from GCS and returns its content as bytes."""
    blob = bucket.blob(file_name)
    file_data = blob.download_as_bytes()  # Fetch the file as bytes
    
    return file_data  # Return the file content as bytes
