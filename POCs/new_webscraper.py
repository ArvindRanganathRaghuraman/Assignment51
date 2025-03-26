import os
import requests
import asyncio
from dotenv import load_dotenv
from playwright.async_api import async_playwright

# Load AWS credentials from .env file
load_dotenv()

# NVIDIA Financial Reports URL
URL = "https://investor.nvidia.com/financial-info/financial-reports/default.aspx"

# Define years and quarters to scrape
YEARS_TO_SCRAPE = ["2022"]
QUARTERS = ["First Quarter"]

# Define the local directory to store PDFs
LOCAL_STORAGE_DIR = "nvidia_financial_reports"

# Create the local directory if it doesn't exist
if not os.path.exists(LOCAL_STORAGE_DIR):
    os.makedirs(LOCAL_STORAGE_DIR)

async def get_pdf_links(page, year, quarter):
    """Scrapes 10-K and 10-Q PDFs for a given year and quarter."""
    print(f"🔍 Scraping {quarter} {year}...")

    # **Select the year in the dropdown**
    dropdown = page.locator("select")
    await dropdown.wait_for(state="visible", timeout=30000)
    await dropdown.select_option(year)
    await page.wait_for_load_state("networkidle")  # Ensure reload completes

    # **Expand the quarter section**
    quarter_section = page.locator(f"text={quarter}")
    plus_button = quarter_section.locator("xpath=..").locator("button")

    if await plus_button.is_visible():
        await plus_button.click()
        await page.wait_for_load_state("networkidle")

    # **Find `10-K` and `10-Q` Links**
    pdf_links = []
    for report_type in ["10-K", "10-Q"]:
        report_locator = page.locator(f"text={report_type}")

        if await report_locator.is_visible():
            print(f"📄 Clicking {report_type} for {quarter} {year}...")

            # **Use JavaScript to force-click**
            await page.evaluate('(element) => element.click()', await report_locator.element_handle())
            await page.wait_for_load_state("networkidle")

            # **Extract PDF URL**
            pdf_url = await page.evaluate('''() => {
                let link = document.querySelector("a[href$='.pdf']");
                return link ? link.href : null;
            }''')

            if pdf_url:
                pdf_links.append(pdf_url)
                print(f"✅ Found {report_type} PDF: {pdf_url}")

    return pdf_links

async def download_pdfs_locally(pdf_links, year, quarter):
    """Downloads PDFs and saves them locally under `nvidia_financial_reports/{YEAR}/{QUARTER}/`."""
    local_quarter_dir = os.path.join(LOCAL_STORAGE_DIR, year, quarter.replace(' ', '_'))
    if not os.path.exists(local_quarter_dir):
        os.makedirs(local_quarter_dir)

    for pdf_url in pdf_links:
        pdf_name = pdf_url.split("/")[-1]

        print(f"📥 Downloading {pdf_name} for {year} - {quarter}...")
        response = requests.get(pdf_url, stream=True)
        if response.status_code == 200:
            local_pdf_path = os.path.join(local_quarter_dir, pdf_name)
            with open(local_pdf_path, 'wb') as f:
                f.write(response.content)
            print(f"✅ Saved {pdf_name} locally in {local_pdf_path}")
        else:
            print(f"❌ Failed to download {pdf_url}")

async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)  # Debug mode to see interactions
        page = await browser.new_page()
        
        print("🚀 Opening webpage...")
        await page.goto(URL, timeout=90000)
        await page.wait_for_load_state("networkidle")

        for year in YEARS_TO_SCRAPE:
            for quarter in QUARTERS:
                pdf_links = await get_pdf_links(page, year, quarter)
                if pdf_links:
                    await download_pdfs_locally(pdf_links, year, quarter)
                else:
                    print(f"❌ No PDFs found for {quarter} {year}.")

        await browser.close()

# Run the scraper
asyncio.run(main())
