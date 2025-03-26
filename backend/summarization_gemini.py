import os
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv(dotenv_path=".env")
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("Error: GEMINI_API_KEY is missing! Check your .env file.")

# Configure Google Gemini API
genai.configure(api_key=api_key)

def summarize_text_gemini(text):
    """Use Google Gemini 2.0 Flash to summarize extracted text."""
    try:
        model = genai.GenerativeModel("gemini-2.0-flash")  # ✅ Use Gemini 2.0 Flash
        response = model.generate_content(
            f"Summarize the following document:\n\n{text}"
        )

        # ✅ Correct way to extract text from the response
        if response and hasattr(response, "candidates") and response.candidates:
            # Extract the first candidate's content
            candidate = response.candidates[0]
            
            # Check if it has 'content' and 'parts'
            if hasattr(candidate, "content") and hasattr(candidate.content, "parts"):
                return candidate.content.parts[0].text  # ✅ Extract text properly
            
        return "Error: Unexpected response format from Gemini API."

    except Exception as e:
        return f"Error: {e}"
