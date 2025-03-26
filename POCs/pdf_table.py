import pdfplumber
import pandas as pd
import os

# Input PDF file
pdf_path = "/Users/riyasingh/Downloads/ilovepdf_split/arxiv_sample-5.pdf"
output_folder = "extracted_tables"

# Create output directory if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Function to save tables as CSV
def save_table(data, filename):
    df = pd.DataFrame(data)
    df.to_csv(os.path.join(output_folder, filename), index=False)


print("Extracting tables using pdfplumber...")
with pdfplumber.open(pdf_path) as pdf:
    for page_num, page in enumerate(pdf.pages):
        tables = page.extract_tables()
        for table_index, table in enumerate(tables):
            save_table(table, f"pdfplumber_page{page_num+1}_table{table_index+1}.csv")

print(f"Tables saved in '{output_folder}'")
