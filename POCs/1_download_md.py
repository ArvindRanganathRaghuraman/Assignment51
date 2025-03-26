from google.cloud import storage
import json
import os

def download_md_file(bucket_name, prefix, file_name):
    """Downloads a specific .md file from GCS and saves it back in the 'rags/' folder."""
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(f"{prefix}{file_name}")

    if not blob.exists():
        print(f"File {file_name} not found in {prefix}")
        return

    # Download the file content
    content = blob.download_as_text()

    # Save the file content locally as JSON
    json_file_name = file_name.replace('.md', '.json')
    local_json_path = f"/tmp/{json_file_name}"  # Temporary storage before uploading

    with open(local_json_path, "w") as f:
        json.dump({file_name: content}, f)

    # Upload the JSON file to 'rags/' folder in the same bucket
    rags_blob = bucket.blob(f"rags/{json_file_name}")
    rags_blob.upload_from_filename(local_json_path)

    print(f"File {file_name} processed and saved as rags/{json_file_name} in GCS.")

if __name__ == "__main__":
    bucket_name = "pdfstorage_1"
    prefix = "outputs/"
    file_name = "PDF_File(1).pdf.md"  # Replace with the desired file name

    download_md_file(bucket_name, prefix, file_name)
