import pytesseract

pytesseract.pytesseract.tesseract_cmd = r''

str = pytesseract.image_to_string()



import os
from google.cloud import vision
import fitz  # PyMuPDF

# Set the environment variable for authentication
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "YOUR_JSON_CREDENTIALS_FILE.json"


def extract_images_as_bytes(pdf_path):
    doc = fitz.open(pdf_path)
    images_bytes = []
    for page_num in range(2):  # First two pages
        page = doc.load_page(page_num)
        for img_index, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            images_bytes.append(image_bytes)
    return images_bytes


def perform_ocr_on_images(images_bytes):
    client = vision.ImageAnnotatorClient()
    texts = []
    for image_bytes in images_bytes:
        image = vision.Image(content=image_bytes)
        response = client.text_detection(image=image)
        texts.append(response.text_annotations[0].description if response.text_annotations else "")
    return texts


def extract_text_from_first_two_pages(pdf_path):
    images_bytes = extract_images_as_bytes(pdf_path)
    extracted_texts = perform_ocr_on_images(images_bytes)
    return "\n".join(extracted_texts)

pdf_path = 'E:\\FlaskMongo old\\Book\\Books\\The World According to Garp.pdf'  # Replace with your PDF file path
extracted_text = extract_text_from_first_two_pages(pdf_path)
print(extracted_text)
