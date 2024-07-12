from flask import Flask, send_file
from bson import ObjectId
from pymongo import MongoClient
from gridfs import GridFS

app = Flask(__name__)

# Connect to MongoDB
client = MongoClient("mongodb+srv://zain:mirha456@cluster0.hog4kvb.mongodb.net/")
db = client["myDatabase"]
fs = GridFS(db, collection="Books")  # Specify the collection name "Books"

@app.route('/open_book/<file_id>')
def open_book(file_id):
    # Retrieve the file from GridFS using the provided file_id
    pdf_file = fs.get(ObjectId(file_id))

    if pdf_file:
        # Save the PDF file to a temporary location
        temp_path = f'tmp_{file_id}.pdf'
        with open(temp_path, 'wb') as temp_file:
            temp_file.write(pdf_file.read())

        # Send the file as a response to the browser
        return send_file(temp_path, as_attachment=True, download_name='Free - Paul Vincent.pdf')
    else:
        return 'Book not found', 404

if __name__ == '__main__':
    app.run(debug=True, port=5002)
