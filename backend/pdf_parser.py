import fitz  # PyMuPDF
from markdownify import markdownify as md
import re
import io
from fastapi import UploadFile
from gcs_utils import upload_to_gcs  # Import the upload function

def extract_and_remove_links(text):
    """Extracts all links from the text and removes them from the original content."""
    url_pattern = r"https?://[^\s]+"  
    links = re.findall(url_pattern, text) 
    cleaned_text = re.sub(url_pattern, "", text)
    return cleaned_text, links

async def pdf_to_markdown(file: UploadFile):
    """Extracts text from an uploaded PDF, removes inline links, and uploads as Markdown to GCS."""
    # Read PDF file content into memory
    pdf_bytes = await file.read()
    
    # Open the PDF in-memory using PyMuPDF
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    extracted_text = ""
    all_links = []

    for page in doc:
        page_text = page.get_text("text")
        cleaned_text, links = extract_and_remove_links(page_text)
        extracted_text += cleaned_text + "\n\n"
        all_links.extend(links)

    markdown_text = md(extracted_text)

    if all_links:
        markdown_text += "\n\n## References\n"
        for i, link in enumerate(set(all_links), start=1):  
            markdown_text += f"{i}. {link}\n"

    # Create an in-memory Markdown file
    markdown_bytes = io.BytesIO(markdown_text.encode("utf-8"))
    
    # Define GCS path: store in `outputs/` inside the GCS bucket
    md_filename = f"outputs/{file.filename}.md"

    # Upload directly to GCS from memory
    gcs_file_url = upload_to_gcs(markdown_bytes, md_filename)

    return {"gcs_url": gcs_file_url}
