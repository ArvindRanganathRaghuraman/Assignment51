# In your terminal, first run:
# pip install openai

import os
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv(dotenv_path=".env")

# Set up Groq API client
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

client = OpenAI(
    api_key=GROQ_API_KEY,
    base_url="https://api.x.ai/v1",  # ✅ Groq API OpenAI-compatible base URL
)

def summarize_text_groq(text):
    """Use Groq API via OpenAI SDK to summarize extracted text."""
    try:
        completion = client.chat.completions.create(
            model="grok-2",  # ✅ Using Groq-supported Mixtral model
            messages=[
                {"role": "system", "content": "You are an AI that summarizes documents concisely."},
                {"role": "user", "content": f"Summarize the following document:\n\n{text}"}
            ],
            temperature=0.7
        )

        return completion.choices[0].message.content  # ✅ Extract and return summary

    except Exception as e:
        return f"Error: {e}"


