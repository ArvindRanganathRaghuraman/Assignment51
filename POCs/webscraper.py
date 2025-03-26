import os
import requests
import asyncio
from dotenv import load_dotenv
from playwright.async_api import async_playwright

# Load environment variables
load_dotenv()

# NVIDIA Financial Reports Page
URL = "https://investor.nvidia.com/financial-info/financial-reports/default.aspx"

# Directory to save downloaded PDFs locally
DOWNLOAD_DIR = "downloaded_pdfs"

# Create the download directory if it doesn't exist
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

async def get_pdf_links():
    """Extracts 10-Q and 10-K PDF links from NVIDIA's financial page using Playwright."""
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)  # Debug mode enabled
        page = await browser.new_page()

        print("🚀 Loading webpage...")
        await page.goto(URL, timeout=90000)
        await page.wait_for_load_state("networkidle")  # Ensure page loads completely

        print("🔍 Extracting all links on page...")
        all_links = await page.evaluate('''() => {
            return Array.from(document.querySelectorAll("a")).map(a => ({
                text: a.innerText,
                href: a.href
            }));
        }''')

        # Filter only 10-Q and 10-K PDFs
        pdf_links = [link["href"] for link in all_links if ".pdf" in link["href"] and ("10-Q" in link["text"] or "10-K" in link["text"])]

        await browser.close()

    return pdf_links

async def download_pdfs_locally(pdf_links):
    """Downloads PDFs and saves them locally."""
    for pdf_url in pdf_links:
        pdf_name = pdf_url.split("/")[-1]
        local_path = os.path.join(DOWNLOAD_DIR, pdf_name)

        print(f"📥 Downloading {pdf_name}...")
        response = requests.get(pdf_url, stream=True)
        if response.status_code == 200:
            # Save the PDF locally
            with open(local_path, "wb") as f:
                f.write(response.content)
            print(f"✅ Saved {pdf_name} to {local_path}")
        else:
            print(f"❌ Failed to download {pdf_url}")

async def main():
    pdf_links = await get_pdf_links()
    if pdf_links:
        print(f"✅ Found {len(pdf_links)} PDF links. Downloading locally...")
        await download_pdfs_locally(pdf_links)
    else:
        print("❌ No PDFs found.")

# Run the scraper
asyncio.run(main())
