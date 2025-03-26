from fastapi import FastAPI, File, UploadFile, Body
from urllib.parse import unquote
from pdf_parser import pdf_to_markdown
from gcs_utils import list_files_in_gcs, get_file_content, download_file_from_gcs
from summarization_gpt import summarize_text_gpt
from summarization_gemini import summarize_text_gemini
from summarization_deepseek import summarize_text_deepseek
from summarization_claude import summarize_text_claude
from backend.rag_qa import answer_question_gpt  # ✅ Uses Pinecone for retrieval & GPT-4o for response
from pinecone_indexing import index_markdown_data  # ✅ Index Markdown in Pinecone

app = FastAPI()

# Model mapping dictionaries
SUMMARIZATION_MODELS = {
    "gpt": summarize_text_gpt,
    "gemini": summarize_text_gemini,
    "deepseek": summarize_text_deepseek,
    "claude": summarize_text_claude
}

@app.get("/")
def read_root():
    return {"message": "Welcome to the FastAPI PDF Processing & Q/A Service"}

@app.post("/upload_pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    """Handles PDF uploads, converts them to Markdown, and uploads to GCS."""
    markdown_info = await pdf_to_markdown(file)
    return {"message": "File processed successfully", "gcs_url": markdown_info["gcs_url"]}

@app.get("/list_files/")
def list_files():
    """Lists all files in the GCS bucket."""
    return {"files": list_files_in_gcs()}

@app.get("/get_file/{file_name:path}")
def get_file(file_name: str):
    """Fetches Markdown content from GCS, indexes it in Pinecone, and allows Q/A on it."""
    decoded_file_name = unquote(file_name)
    try:
        content = get_file_content(decoded_file_name)
        if not content:
            return {"error": "File content is empty"}

        # ✅ Debugging: Print the Markdown content before indexing
        print(f"Retrieved Markdown Content for {decoded_file_name}: \n{content}")

        # ✅ Index the Markdown content into Pinecone before allowing Q/A
        index_markdown_data(content, decoded_file_name)

        return {
            "file_name": decoded_file_name,
            "content": content,  # ✅ Ensure this is returned in API response
            "message": "Markdown content retrieved and indexed successfully. You can now ask questions."
        }
    
    except Exception as e:
        return {"error": f"Failed to fetch and index file: {str(e)}"}

@app.get("/download_file/{file_name:path}")
def download_file(file_name: str):
    """Provides a file for download from Google Cloud Storage."""
    decoded_file_name = unquote(file_name)
    return download_file_from_gcs(decoded_file_name)

@app.post("/summarize_file/")
def summarize_file(content: str = Body(..., embed=True), model: str = Body("gpt", embed=True)):
    """Summarizes the given text using the selected model."""
    try:
        if not content.strip():
            return {"error": "File content is empty"}
        
        if model not in SUMMARIZATION_MODELS:
            return {"error": f"Invalid model '{model}'. Choose from {list(SUMMARIZATION_MODELS.keys())}"}
        
        # Call the appropriate summarization function
        summarization_function = SUMMARIZATION_MODELS[model]
        summary = summarization_function(content)

        return {"summary": summary}
    
    except Exception as e:
        return {"error": f"Failed to summarize file: {str(e)}"}

@app.post("/ask_question/")
def ask_question(question: str = Body(..., embed=True)):
    """Handles Q/A on the selected Markdown using Pinecone & GPT-4o."""
    try:
        if not question.strip():
            return {"error": "Question cannot be empty"}

        # ✅ Use Pinecone + GPT-4o for answering questions based on Markdown content
        answer = answer_question_gpt(question)

        return {"question": question, "answer": answer}
    
    except Exception as e:
        return {"error": f"Failed to process question: {str(e)}"}
