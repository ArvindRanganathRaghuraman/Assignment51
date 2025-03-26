import os
import fitz  # PyMuPDF
import cv2
from pdf2image import convert_from_path


# Input PDF file
pdf_path = "/Users/riyasingh/Desktop/arxiv_sample.pdf"
output_folder = "extracted_images"

# Create output directory if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Function to save image data
def save_image(image_data, filename):
    with open(os.path.join(output_folder, filename), "wb") as f:
        f.write(image_data)

# 1. Extract Images Using PyMuPDF (fitz)
print("Extracting embedded images using PyMuPDF...")
doc = fitz.open(pdf_path)
found_images = False

for page_num, page in enumerate(doc):
    for img_index, img in enumerate(page.get_images(full=True)):
        found_images = True
        xref = img[0]
        base_image = doc.extract_image(xref)
        img_data = base_image["image"]
        img_ext = base_image["ext"]
        save_image(img_data, f"fitz_page{page_num+1}_img{img_index+1}.{img_ext}")

doc.close()

# 2. If no embedded images found, render pages as images
if not found_images:
    print("No embedded images found. Converting PDF pages to images...")

    # Convert each page to an image
    images = convert_from_path(pdf_path, dpi=300)

    for i, image in enumerate(images):
        img_path = os.path.join(output_folder, f"rendered_page{i+1}.png")
        image.save(img_path, "PNG")
        print(f"Saved rendered image: {img_path}")

        # Process the image to extract individual images inside the rendered page
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

        # Find contours of possible images
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for j, contour in enumerate(contours):
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filter out small contours that are unlikely to be images
            if w > 50 and h > 50:
                cropped_img = img[y:y+h, x:x+w]
                cropped_img_path = os.path.join(output_folder, f"detected_img_page{i+1}_{j+1}.png")
                cv2.imwrite(cropped_img_path, cropped_img)
                print(f"Extracted possible image: {cropped_img_path}")

print(f"Images saved in '{output_folder}'")
