import os
import time
import requests
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import Select
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from io import BytesIO
from backend.gcs_utils import upload_to_gcs  # Importing GCS upload function

# Configuration
BASE_URL = "https://investor.nvidia.com/financial-info/quarterly-results/default.aspx"

def download_pdf_to_gcs(url, gcs_path):
    """
    Downloads a PDF from the given URL and uploads it directly to GCS.
    """
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise an exception for 4xx/5xx errors

        # Wrap bytes in BytesIO to create a file-like object
        pdf_bytes = BytesIO(response.content)
        pdf_bytes.seek(0)  # Ensure pointer is at the beginning

        # Upload the PDF to GCS
        gcs_file_url = upload_to_gcs(pdf_bytes, gcs_path)
        print(f"✅ Uploaded to GCS: {gcs_file_url}")

    except requests.exceptions.RequestException as e:
        print(f"❌ Error downloading {url}: {str(e)}")

def get_nvidia_quarterly_pdfs(year):
    """
    Scrapes NVIDIA's investor relations page for 10-K/10-Q reports for a given year
    and uploads the PDFs to GCS.
    """
    print(f"📌 Starting NVIDIA financial report scraper for year: {year}")

    # Selenium options for headless Chrome
    chrome_options = Options()
    chrome_options.add_argument("--headless=new")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--window-size=1920,1080")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/112.0.0.0 Safari/537.36"
    )

    driver = webdriver.Chrome(
        service=Service(ChromeDriverManager().install()), options=chrome_options
    )

    try:
        print(f"🌍 Accessing {BASE_URL}")
        driver.get(BASE_URL)
        time.sleep(5)  # Allow time for the page to load

        # Select the year from the dropdown
        try:
            year_dropdown = driver.find_element(By.ID, "_ctrl0_ctl75_selectEvergreenFinancialAccordionYear")
            year_select = Select(year_dropdown)
            year_select.select_by_value(str(year))
            print(f"✅ Selected year {year} from dropdown")
            time.sleep(3)
        except Exception as e:
            print(f"❌ Error selecting year {year}: {str(e)}")

        # Locate all quarter accordion items
        accordion_items = driver.find_elements(By.CSS_SELECTOR, "div.evergreen-accordion.evergreen-financial-accordion-item")
        print(f"🔎 Found {len(accordion_items)} quarter sections on the page")

        for item in accordion_items:
            try:
                # Expand the accordion if not already expanded
                try:
                    toggle_button = item.find_element(By.CSS_SELECTOR, "button.evergreen-financial-accordion-toggle")
                    if toggle_button.get_attribute("aria-expanded") == "false":
                        toggle_button.click()
                        time.sleep(1)
                except Exception:
                    print("⚠️ Could not expand accordion item, possibly already expanded")

                # Extract the quarter title (e.g., "Fourth Quarter 2025")
                title_elem = item.find_element(By.CSS_SELECTOR, "span.evergreen-accordion-title")
                quarter_text = title_elem.text.strip()
                print(f"📁 Processing: {quarter_text}")

                # Determine quarter name
                quarter = None
                if "Fourth Quarter" in quarter_text:
                    quarter = "Q4"
                elif "Third Quarter" in quarter_text:
                    quarter = "Q3"
                elif "Second Quarter" in quarter_text:
                    quarter = "Q2"
                elif "First Quarter" in quarter_text:
                    quarter = "Q1"
                else:
                    print(f"⚠️ Could not determine quarter from title: {quarter_text}")
                    continue

                # Extract all PDF links for 10-K and 10-Q reports
                pdf_links = item.find_elements(By.CSS_SELECTOR, "a.evergreen-financial-accordion-attachment-PDF")
                print(f"🔗 Found {len(pdf_links)} PDF links in {quarter_text}")

                for link in pdf_links:
                    href = link.get_attribute("href")
                    if not href or not href.endswith(".pdf"):
                        continue

                    # Extract text or aria-label
                    link_text = ""
                    try:
                        span = link.find_element(By.CSS_SELECTOR, "span.evergreen-link-text.evergreen-financial-accordion-link-text")
                        link_text = span.text.strip()
                    except Exception:
                        link_text = link.get_attribute("aria-label") or ""

                    # Check if it's a 10-K or 10-Q report
                    if "10-k" in link_text.lower() or "10-q" in link_text.lower():
                        # GCS file path format: "nvidia_reports/{year}/{quarter}/{filename}.pdf"
                        gcs_filename = f"pdf_files/{year}/{quarter}/{link_text.replace(' ', '_').replace('/', '_')}.pdf"
                        print(f"📥 Uploading to GCS: {gcs_filename}")

                        # Download and upload PDF to GCS
                        download_pdf_to_gcs(href, gcs_filename)
            except Exception as e:
                print(f"❌ Error processing quarter accordion: {str(e)}")

        print(f"📂 Completed uploading PDFs for {year}")

    except Exception as e:
        print(f"❌ Error scraping NVIDIA reports for {year}: {str(e)}")
    finally:
        driver.quit()

# Run the scraper for multiple years (2021–2025)
if __name__ == "__main__":
    years_to_scrape = range(2021, 2026)  # Scrape from 2021 to 2025
    for year in years_to_scrape:
        print(f"🚀 Scraping NVIDIA reports for {year}...")
        get_nvidia_quarterly_pdfs(year)
        time.sleep(2)  # Small delay to avoid hitting the server too fast
