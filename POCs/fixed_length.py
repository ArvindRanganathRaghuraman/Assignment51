import PyPDF2

# Function to read PDF and return text
def read_pdf(pdf_path):
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        text = ""
        for page in range(len(reader.pages)):
            text += reader.pages[page].extract_text()
    return text

# Function to chunk text manually into fixed-length chunks
def fixed_length_chunking(pdf_path, chunk_size=512):
    text = read_pdf(pdf_path)
    
    # Split the text into chunks of fixed length
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    
    return chunks

# Example usage:
pdf_path = "10K10Q-Quarter-1-2025.pdf"
chunks = fixed_length_chunking(pdf_path)

for chunk in chunks:
    print(chunk)
