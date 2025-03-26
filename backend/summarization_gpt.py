import os
from litellm import completion
from dotenv import load_dotenv

# Load environment variables
load_dotenv(dotenv_path=".env")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

def summarize_text_gpt(text):
    """Use GPT-4o Mini to summarize extracted text from a PDF."""
    response = completion(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are an AI that summarizes documents concisely."},
            {"role": "user", "content": f"Summarize the following document:\n\n{text}"}
        ]
    )
    return response['choices'][0]['message']['content']

'''
def answer_question_gpt(text, question):
    """Use GPT-4o Mini to answer questions based on extracted text from a PDF."""
    response = completion(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are an AI assistant that answers questions based on the given document."},
            {"role": "user", "content": f"Document:\n{text}\n\nAnswer the following question based ONLY on the document above:\n{question}"}
        ]
    )
    return response['choices'][0]['message']['content']'
'''
