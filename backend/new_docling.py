import logging
import time
from pathlib import Path
from docling_core.types.doc import ImageRefMode
from docling.document_converter import DocumentConverter
from gcs_utils import upload_to_gcs
from io import BytesIO

def process_pdf(pdf_path, output_dir, gcs_output_bucket="pdfstorage_1"):
    """Process a single PDF file, save it as Markdown, and upload to GCS."""
    logging.basicConfig(level=logging.INFO)

    input_doc_path = Path(pdf_path)
    output_dir = Path(output_dir)

    # Initialize the DocumentConverter
    doc_converter = DocumentConverter()

    start_time = time.time()
    try:
        # Convert the document
        conv_res = doc_converter.convert(input_doc_path)

        # Ensure the output directory exists
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save markdown with embedded images
        md_filename_embedded = output_dir / f"{input_doc_path.stem}-with-images.md"
        conv_res.document.save_as_markdown(md_filename_embedded, image_mode=ImageRefMode.EMBEDDED)
        

        # Upload the Markdown file to GCS using BytesIO
        with open(md_filename_embedded, 'rb') as file_embedded:
            file_stream_embedded = BytesIO(file_embedded.read())  # Convert the file to BytesIO

            # Upload to GCS
            gcs_file_url = upload_to_gcs(file_stream_embedded, f"outputs/{md_filename_embedded.stem}.md")
            

        # Clean up local markdown file after upload (if needed)
        md_filename_embedded.unlink()

        end_time = time.time() - start_time
        logging.info(f"Document converted and uploaded in {end_time:.2f} seconds.")

        success_message = f"Document successfully converted and uploaded to GCS"
        print(success_message)  # This will print in the server logs

        # Return success message
        return {"message": success_message}
        

    except Exception as e:
        logging.error(f"Failed to process {pdf_path}: {e}")
        return None  # Return None in case of failure
