import fitz  # PyMuPDF
import pytesseract
from PIL import Image

# Set the path to the Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# PDF file path
pdf_path = "E:\\FlaskMongo old\\Book\\Books\\siddharthaaa.pdf"


# Function to check if the page contains an image
def is_page_image(page):
    images = page.get_images()
    return len(images) > 0

# Function to extract text from a page
def extract_text(page):
    return page.get_text()

# Function to extract title from text
def extract_title(text):
    # Assuming the title is the first line of the text
    lines = text.split('\n')
    if lines:
        return lines[0]
    return None


# Open the PDF file
pdf_document = fitz.open(pdf_path)

# Get the first page
first_page = pdf_document[0]

# Check if the first page contains an image
if is_page_image(first_page):
    # Render the PDF page as an image
    image = first_page.get_pixmap()
    # Convert image to PIL format
    pil_image = Image.frombytes("RGB", [image.width, image.height], image.samples)
    # Perform OCR on the image and print the result
    string = pytesseract.image_to_string(pil_image)
    print("Extracted Text:", string)
else:
    # Extract text from the page
    text = extract_text(first_page)
    if text:
        # Extract title from the text
        title = extract_title(text)
        if title:
            print("Title:", title)
        else:
            print("No title found on the first page.")
    else:
        print("No text found on the first page.")

# Close the PDF document
pdf_document.close()
