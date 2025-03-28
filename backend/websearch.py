from serpapi import GoogleSearch
from dotenv import load_dotenv 
import os

load_dotenv(dotenv_path=".env")
SERPAPI_KEY = os.getenv("SERPAPI_KEY")  

def fetch_nvidia_news(query="NVIDIA latest news"):
    params = {
        "q": query,
        "location": "Austin, Texas, United States",
        "hl": "en",
        "gl": "us",
        "google_domain": "google.com",
        "tbm": "nws",  # Ensures that only news results are returned
        "api_key": SERPAPI_KEY
    }

    search = GoogleSearch(params)
    results = search.get_dict()

    # Debug: Print full API response
   # print("Full Response:", results)

    # Extract and print news results
    if "news_results" in results:
        news = [f"{item['title']}: {item['link']}" for item in results["news_results"]]
        return "\n".join(news)

    return "No news found."

if __name__ == "__main__":
    print(fetch_nvidia_news())
