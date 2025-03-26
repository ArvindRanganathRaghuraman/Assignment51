import os
from langchain.text_splitter import RecursiveCharacterTextSplitter

def langchain_chunking(md_file_path, chunk_size=512, chunk_overlap=50):
    """
    Reads a Markdown (.md) file and chunks the text using LangChain's RecursiveCharacterTextSplitter.

    Args:
        md_file_path (str): Path to the Markdown file.
        chunk_size (int): The size of each chunk (default: 512).
        chunk_overlap (int): Overlapping tokens between chunks (default: 50).

    Returns:
        list: A list of text chunks.
    """

    
    if not md_file_path.endswith(".md"):
        raise ValueError(" Unsupported file format. This function only processes .md (Markdown) files.")

    
    with open(md_file_path, "r", encoding="utf-8") as f:
        text = f.read()

    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_text(text)

    return chunks

# Example Usage
chunks = langchain_chunking("POCs/DOCLING_PDF_PLUMBER_Markdowns/10K10Q-Q1-2024-with-image-refs.md", chunk_size=512, chunk_overlap=50)
print("\n Markdown Chunks (Preview):", chunks[:3])  # Preview first 3 chunks
