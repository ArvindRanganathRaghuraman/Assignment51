import os
import fitz  # PyMuPDF for extracting text from PDFs
import google.generativeai as genai
from dotenv import load_dotenv

# Load API Key from .env
load_dotenv(dotenv_path=".env")
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("Error: GEMINI_API_KEY is missing! Check your .env file.")


genai.configure(api_key=api_key)

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file."""
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text("text") + "\n"
    return text

def summarize_text_gemini(text):
    """Use Google Gemini to summarize extracted text."""
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")  # ✅ Use Gemini API directly
        response = model.generate_content(
            f"Summarize the following document:\n\n{text}"
        )

        return response.text  # ✅ Extracts and returns the generated summary

    except Exception as e:
        return f"Error: {e}"

def main():
    """Extract text from PDF and summarize using Google Gemini API."""
    pdf_file_path = "arxiv_sample.pdf"  # ✅ Replace with your actual file path

    # Extract text from PDF
    document_text = extract_text_from_pdf(pdf_file_path)

    # Summarize extracted text
    summary = summarize_text_gemini(document_text)

    # Print summary
    print("Summary:\n", summary)

# Run the script
if __name__ == "__main__":
    main()
