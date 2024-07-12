from pymongo import MongoClient
from gridfs import GridFS
import pymongo

client = pymongo.MongoClient("mongodb+srv://zain:mirha456@cluster0.hog4kvb.mongodb.net/myDatabase")
data = client["myDatabase"]
all_books = data["all_books"]

# Connect to MongoDB
client = MongoClient("mongodb://localhost:27017/")
db = client["myDatabase"]
fs = GridFS(db, collection="Books")  # Specify the collection name "Books"

def update_link(book_title, new_link):
    query = {"Book-Title": book_title}
    new_values = {"$set": {"Link": new_link}}
    all_books.update_one(query, new_values)

def store_pdf(pdf_path, filename):
    filenames = filename + ".pdf"
    # Open the PDF file in binary mode
    with open(pdf_path, 'rb') as pdf_file:
        fs.put(pdf_file, filename=filenames)
    print(f"PDF stored in collection 'Books' with ObjectId: {fs.get_last_version(filenames).id}")
    output_str = str({fs.get_last_version(filenames).id})
    ids = output_str[11:-3]
    new_link = "http://127.0.0.1:5001/open_book/" + ids
    # Update the link in the database
    update_link(filename, new_link)
    print("Link updated successfully!")

pdf_path = 'C:\\Users\\M.Qasim\\Desktop\\FlaskMongo\\venv\\Scripts\\Book\\Books\\The World According to Garp.pdf'
filename = input("Enter Book Name: ")
store_pdf(pdf_path, filename)



# Function to update link based on book title

