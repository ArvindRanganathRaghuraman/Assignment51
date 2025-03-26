import os
import fitz  # PyMuPDF
from litellm import completion
from dotenv import load_dotenv



# Set OpenAI API Key
load_dotenv(dotenv_path="/Users/arvindranganathraghuraman/Desktop/Assignment-4-PoC/.env")

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file"""
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text("text") + "\n"
    return text

def summarize_text_gpt(text):
    """Use GPT-4o Mini to summarize the extracted text"""
    response = completion(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are an AI that summarizes documents concisely."},
            {"role": "user", "content": f"Summarize the following document:\n\n{text}"}
        ]
    )
    return response['choices'][0]['message']['content']

# Specify the PDF file path
pdf_file_path = "arxiv_sample.pdf"

# Extract text from PDF
document_text = extract_text_from_pdf(pdf_file_path)

# Summarize extracted text using GPT-4o Mini
summary = summarize_text_gpt(document_text)

# Print summary
print("Summary:\n", summary)
