import io
import json
from mistralai import Mistral, DocumentURLChunk
from dotenv import load_dotenv
import os
from gcs_utils import upload_to_gcs  # Import GCS upload function

# Load API key
load_dotenv()
api_key = os.getenv("MISTRAL_API_KEY")
client = Mistral(api_key=api_key)

def process_pdf_mistral(pdf_url: str):
    """Processes a PDF, extracts OCR data, converts to Markdown, and uploads to GCS."""
    
    print(f"Processing {pdf_url} ...")
    
    # Perform OCR
    ocr_response = client.ocr.process(
        document=DocumentURLChunk(document_url=pdf_url),
        model="mistral-ocr-latest",
        include_image_base64=True
    )

    # Convert OCR pages to Markdown
    markdown_pages = []
    for page in ocr_response.pages:
        markdown_pages.append(page.markdown)
    
    final_markdown = "\n\n".join(markdown_pages)

    # Convert Markdown to bytes for upload
    markdown_bytes = io.BytesIO(final_markdown.encode("utf-8"))

    # Dynamic file name (using the PDF URL name for the markdown file)
    md_filename = f"outputs/{pdf_url.split('/')[-1].replace('.pdf', '.md')}"

    # Upload the Markdown bytes to GCS
    gcs_file_url = upload_to_gcs(markdown_bytes, md_filename)

    print(f"Markdown uploaded to GCS: {gcs_file_url}")
    return {"gcs_url": gcs_file_url}
