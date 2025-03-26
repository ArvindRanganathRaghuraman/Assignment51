import os
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv(dotenv_path=".env")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")


# Define DeepSeek API endpoint
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"

def summarize_text_deepseek(text):
    """Use DeepSeek-Chat to summarize extracted text from a PDF."""
    print("summarize_text_deepseek function invoked") 
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": "You are an AI that summarizes documents concisely."},
            {"role": "user", "content": f"Summarize the following document:\n\n{text}"}
        ],
        "temperature": 0.7
    }

    try:
        response = requests.post(DEEPSEEK_API_URL, headers=headers, json=payload)

        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            return f"DeepSeek API Error: {response.json()}"
    
    except Exception as e:
        return f"DeepSeek Summarization Failed: {str(e)}"
