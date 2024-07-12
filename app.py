from typing import Dict, Union

from flask import Flask, render_template, request, redirect, url_for, send_file, session, jsonify, flash
import numpy as np
from bson import ObjectId
from gridfs import GridFS
from flask_pymongo import PyMongo
import pandas as pd
from werkzeug.security import generate_password_hash, check_password_hash
from sklearn.metrics.pairwise import cosine_similarity
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from werkzeug.utils import secure_filename
import os
import random
import string
import time
import threading
import base64
import fitz


# Global variable to store OTP
otp_code = None

app = Flask(__name__)
app.secret_key = 'your_secret_key'
app.config["MONGO_URI"] = "mongodb+srv://zain:mirha456@cluster0.hog4kvb.mongodb.net/myDatabase"
mongo = PyMongo(app)

# GridFS instance for books
fs = GridFS(mongo.db, collection="Books")


# Load the necessary data from MongoDB for KNN (Search)
Rating = pd.DataFrame(list(mongo.db.Rating.find()))
book_pivot = None
similarity_scores = None
books = Rating.drop_duplicates(subset='Book-Name')




def KNN_in_thread():
    global book_pivot, similarity_scores
    x = Rating.groupby('Username').count()['Rating'] > 50  # (<10 ...51975 users, 100 ...)
    print("x", x.size)
    users = x[x].index  # Boolein indexing
    filtered_rating = Rating[Rating['Username'].isin(users)]
    y = filtered_rating.groupby('Book-Name').count()['Rating'] >= 0  # (0 ...33435 books ...)
    print("y", y.size)
    famous_books = y[y].index
    # famous_books = filtered_rating['Book-Title'].unique()
    final_ratings = filtered_rating[filtered_rating['Book-Name'].isin(famous_books)]
    book_pivot = final_ratings.pivot_table(index='Book-Name', columns='Username', values='Rating')
    book_pivot.fillna(0, inplace=True)  # replace NaN with 0
    similarity_scores = cosine_similarity(book_pivot)
    print(book_pivot.shape)
    print(book_pivot)


thread_KNN = threading.Thread(target=KNN_in_thread)
thread_KNN.start()


# Load the necessary data from MongoDB
all_books = pd.DataFrame(list(mongo.db.all_books.find()))



# made rating out of 5
all_books['Avg-Rating'] = all_books['Avg-Rating'] / 2

@app.route('/')
def index():
    print(session)
    if 'username' in session:
        return redirect('/home')
    return redirect('/login')


@app.route('/check-email', methods=['POST'])
def check_email_and_send_welcome():
    data = request.json
    email = data.get('email')
    username = data.get('username')
    password = data.get('password')
    # Check if the email already exists in MongoDB
    users = mongo.db.users
    existing_email = users.find_one({'email': email})
    existing_username = users.find_one({'username': username})

    if existing_email:
        # Email already exists, return response indicating not valid
        return jsonify({'valid': False, 'message': 'Email already exists in the database'})
    elif existing_username:
        # Username already exists, return response indicating not valid
        return jsonify({'valid': False, 'message': 'Username already exists in the database'})
    else:
        print("Data save in Mongodb")
        hashed_password = generate_password_hash(password)
        users.insert_one({
            'username': username,
            'email': email,
            'password': hashed_password
        })
        try:
            send_welcome_email(email)
            return jsonify({'valid': True, 'messageSent': True})
        except Exception as e:
            print(e)
            return jsonify({'valid': True, 'messageSent': False})


def send_welcome_email(to_email):
    your_email = os.environ.get('EMAIL_ADDRESS')
    your_password = os.environ.get('EMAIL_PASSWORD')

    print("Email Address:", os.environ.get('EMAIL_ADDRESS'))
    print("Email Password:", os.environ.get('EMAIL_PASSWORD'))

    # Setting up the SMTP server
    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.login(your_email, your_password)

    # Creating the email content
    msg = MIMEMultipart()
    msg['From'] = your_email
    msg['To'] = to_email
    msg['Subject'] = "Welcome to Our Service"

    body = "Hi there, welcome to our service! We're glad you're here."
    msg.attach(MIMEText(body, 'plain'))

    # Sending the email
    server.send_message(msg)
    del msg  # Delete the message object after sending
    server.quit()


@app.route('/signup', methods=['GET'])
def signup():
    return render_template('signup.html')


@app.route('/search-choose-category', methods=['GET', 'POST'])
def search_choose_category():
    all_books_ = pd.DataFrame(list(mongo.db.all_books.find()))
    all_books_['Category'] = all_books_['Category'].str.lower()
    if request.method == 'GET':
        category_count = all_books_['Category'].value_counts()
        category_count.index = category_count.index.str.title()

        categories = category_count.index.str.title()
        # categories = [f"{category:<15}  {count}" for category, count in category_count.items()]
        return render_template('choose_category.html', categories=categories)

    query = request.form['search']
    matched_category = []
    if query:
        # Search for book category that match the query
        category_name = all_books_['Category'].drop_duplicates()
        matched_category = [book for book in category_name if query.lower() in book.lower()]
        print(matched_category)

    if not matched_category:
        msg = "No Book Found with this SearchED Category!"
        return render_template('choose_category.html', msg=msg)

    data = []  # List to store book details
    No_of_category = []
    # Search for the book in the Pandas DataFrame
    for user_input in matched_category:
        i = 0
        all_books_['Category'] = all_books_['Category'].str.lower()
        results = all_books_[all_books_['Category'] == user_input].iloc[::-1]
        results['Category'] = results['Category'].apply(lambda x: ' '.join(word.capitalize() for word in x.split()))
        for custom_index, row in results.iterrows():
            i = i + 1
            searched_book = {
                'Book-Title': row['Book-Title'],
                'Category': row['Category'],
                'Book-Author': row['Book-Author'],
                'Image-URL-M': row['Image-URL-M'],
                'votes': row['Num-Rating'],
                'rating': row['Avg-Rating']
            }
            data.append(searched_book)
        No_of_category.append(i)
    print(data)
    # Capitalizing
    matched_category = [x.title() for x in matched_category]
    print(matched_category)
    print(No_of_category)
    return render_template('choose_category.html', data=data,
                           Name_of_category=matched_category, No_of_category=No_of_category)


@app.route('/process_categories', methods=['POST'])
def process_categories():
    all_books_ = pd.DataFrame(list(mongo.db.all_books.find()))
    query = request.form.getlist('selected_categories')

    # Convert category names and query to lowercase
    all_books_['Category'] = all_books_['Category'].str.lower()
    query = [q.lower() for q in query]

    category_name = all_books_['Category'].drop_duplicates()
    matched_category = [book for book in category_name if any(q == book for q in query)]

    data = []  # List to store book details
    No_of_category = []
    # Search for the book in the Pandas DataFrame
    for user_input in matched_category:
        i = 0
        results = all_books_[all_books_['Category'] == user_input].iloc[::-1]
        print(results.index)
        for custom_index, row in results.iterrows():
            i = i + 1
            searched_book = {
                'Book-Title': row['Book-Title'],
                'Category': row['Category'].title(),
                'Book-Author': row['Book-Author'],
                'Image-URL-M': row['Image-URL-M'],
                'votes': row['Num-Rating'],
                'rating': row['Avg-Rating'] / 2,
                'Link': row['Link']
            }
            data.append(searched_book)
        No_of_category.append(i)

    print(data)
    matched_category = [category.title() for category in matched_category]
    print(matched_category)
    print(No_of_category)
    return render_template('choose_category.html', data=data,
                           Name_of_category=matched_category, No_of_category=No_of_category)


@app.route('/search', methods=['POST'])
def search_books():
    query = request.form['search']
    matched_book = []
    matched_author = []
    if query:
        # Search for books that match the query
        matched_book = [book for book in all_books['Book-Title'] if query.lower() in book.lower()]
        matched_author = [book for book in all_books['Book-Author'] if query.lower() in book.lower()]
        print("Match Titles", matched_book)
        print("Match Authors", matched_author)
        matched_author = list(set(matched_author))  # drop duplicate

    if not matched_book:
        if matched_author:
            data = []  # List to store book details
            print("matched_author", matched_author)
            # Search for the book in the Pandas DataFrame
            if len(matched_author) == 1:
                books_by_authors = mongo.db.all_books.find({"Book-Author": matched_author[0]}, {"Book-Title": 1})
            else:
                books_by_authors = mongo.db.all_books.find({"Book-Author": {"$in": matched_author}}, {"Book-Title": 1})
            book_titles = [book["Book-Title"] for book in books_by_authors]
            print("Book name against Authors are: ", book_titles)
            for user_input in book_titles:
                result = all_books[all_books['Book-Title'] == user_input]
                if not result.empty:
                    searched_book = {
                        'Book-Title': result['Book-Title'].values[0],
                        'Book-Author': result['Book-Author'].values[0],
                        'Image-URL-M': result['Image-URL-M'].values[0],
                        'votes': result['Num-Rating'].values[0],
                        'rating': result['Avg-Rating'].values[0]
                    }
                    data.append(searched_book)
            print(data)
            return render_template('index.html', data=data)

        msg = "No Book and Author Found with this Search!"
        return render_template('index.html', msg=msg)

    data = []  # List to store book details
    # Search for the book in the Pandas DataFrame
    for user_input in matched_book:
        result = all_books[all_books['Book-Title'] == user_input]
        if not result.empty:
            searched_book = {
                'Book-Title': result['Book-Title'].values[0],
                'Book-Author': result['Book-Author'].values[0],
                'Image-URL-M': result['Image-URL-M'].values[0],
                'votes': result['Num-Rating'].values[0],
                'rating': result['Avg-Rating'].values[0]
            }
            data.append(searched_book)
    print(data)
    Search_Book_names = [entry['Book-Title'] for entry in data]
    print("Searched books: ", Search_Book_names)

    # Recommend using Collaborative filtering (KNN Algorithm)
    if similarity_scores is not None:
        users_recommendations(Search_Book_names)

    return render_template('index.html', data=data)


def users_recommendations(book):
    username = session['username']
    users = mongo.db.users
    recommended_books = []
    print(book)
    if not isinstance(book, list):
        book = [book]
    for book_name in book:
        if book_name in book_pivot.index:
            # index fetch
            index = np.where(book_pivot.index == book_name)[0][0]
            similar_items = sorted(list(enumerate(similarity_scores[index])), key=lambda x: x[1], reverse=True)[1:5]
            searched_book = []
            temp_df = books[books['Book-Name'] == book_name]
            searched_book.extend(temp_df.drop_duplicates('Book-Name')['Book-Name'].to_list())

            for i in similar_items:
                temp_df = books[books['Book-Name'] == book_pivot.index[i[0]]]
                recommended_books.extend(temp_df.drop_duplicates('Book-Name')['Book-Name'].to_list())
        else:
            print(f"Book '{book_name}' not found in the dataset.")

    # Remove duplicates from recommended_books
    recommended_books = list(set(recommended_books))
    print(recommended_books)
    for rec in recommended_books:
        users.update_one({'username': username}, {'$addToSet': {'Recommendations': rec}})


def users_fav_recommendation(book_title):
    username = session['username']
    users = mongo.db.users
    all_books_ = pd.DataFrame(list(mongo.db.all_books.find()))
    print(book_title)
    user_book_category = all_books_.loc[all_books_['Book-Title'] == book_title, 'Category'].iloc[0]
    same_category_books = all_books_[all_books_['Category'] == user_book_category]
    new_df = same_category_books[['Book-Title', 'Avg-Rating']]

    if new_df.shape[0] < 5:
        print("less than 5: ", new_df)
        for index, row in new_df.iterrows():
            users.update_one({'username': username}, {'$addToSet': {'Fav_Recommendations': row['Book-Title']}})
        return

    user_book_rating = new_df.loc[new_df['Book-Title'] == book_title, 'Avg-Rating'].iloc[0]

    # Drop the user book from the new DataFrame
    new_df = new_df[new_df['Book-Title'] != book_title]

    # Calculate Euclidean distance and save it in a new column
    new_df['Distance'] = np.abs(new_df['Avg-Rating'] - user_book_rating)

    # Sort the DataFrame based on the calculated distances
    new_df_sorted = new_df.sort_values(by='Distance')

    # Print the sorted DataFrame
    print("Books sorted by their similarity to the user's book (ascending order of distance):")
    for index, row in new_df_sorted.head(4).iterrows():
        print("Book Title:", row['Book-Title'])
        rec_fav_book = row['Book-Title']
        print("Average Rating:", row['Avg-Rating'])
        print("Distance from user's book rating:", row['Distance'])

        users.update_one({'username': username}, {'$addToSet': {'Fav_Recommendations': rec_fav_book}})


@app.route('/login', methods=['GET', 'POST'])
def login():
    error_message = None
    if request.method == 'POST':
        # admin check
        if request.form['username'] == 'admin':
            if request.form['password'] == 'qas123':
                return redirect('/admin-choice')

        users = mongo.db.users
        login_user = users.find_one({'username': request.form['username']})

        if login_user:  # Check if user exists with the provided username
            if check_password_hash(login_user['password'], request.form['password']):
                session['username'] = request.form['username']
                return redirect(url_for('index'))
            else:
                error_message = 'Invalid password'
        else:
            # If no user found with the provided username, try with email
            login_user_by_email = users.find_one({'email': request.form['username']})
            if login_user_by_email:
                if check_password_hash(login_user_by_email['password'], request.form['password']):
                    session['username'] = login_user_by_email['username']
                    return redirect(url_for('index'))
                else:
                    error_message = 'Invalid password'
            else:
                error_message = 'Invalid username/email'

    return render_template('login.html', error=error_message or '')


@app.route('/logout')
def logout():
    session.clear()  # Clear the session
    session.pop('username', None)
    return redirect(url_for('login'))


@app.route('/admin-choice', methods=['GET', 'POST'])
def add_upload_book():
    if request.method == 'POST':
        num = None
        if request.method == 'POST':
            print(request.form.get('action'))
            if request.form.get('action') == 'Add a Book':
                num = 1
            elif request.form.get('action') == 'Upload Book PDF':
                num = 2
            elif request.form.get('action') == 'Delete Book':
                num = 3
            else:
                return 'Invalid action'
        return render_template('admin.html', num=num)
    return render_template('admin_choice.html')


@app.route('/upload', methods=['POST'])
def upload_book():
    all_latest_books = pd.DataFrame(list(mongo.db.all_books.find()))
    if 'book' not in request.files:
        return 'No file part'

    book_file = request.files['book']

    if book_file.filename == '':
        return 'No selected file'

    # Encode the PDF data to base64
    pdf_data_base64 = base64.b64encode(book_file.read()).decode('utf-8')

    # Decode the base64 PDF data back to binary
    pdf_data_binary = base64.b64decode(pdf_data_base64)

    # Optionally, save the decoded PDF to a file
    with open("uploaded_book.pdf", "wb") as pdf_file:
        pdf_file.write(pdf_data_binary)

    # Load the PDF data using PyMuPDF (fitz)
    pdf_document = fitz.open(stream=pdf_data_binary, filetype="pdf")

    # PyMuPDF (fitz) provides a straightforward way to access a PDF metadata
    # Extract metadata
    metadata = pdf_document.metadata
    title = metadata.get('title', 'No title found')
    print("Metadata:", metadata)
    print("Title:", title)

    query = title
    matched_book = []
    matched_author = []
    if query:
        # Search for books that match the query
        matched_book = [book for book in all_latest_books['Book-Title'] if query.lower() in book.lower()]
        matched_author = [book for book in all_latest_books['Book-Author'] if query.lower() in book.lower()]
        print("Match Titles", matched_book)
        print("Match Authors", matched_author)
        matched_author = list(set(matched_author))  # drop duplicate

    data = []  # List to store book details
    # Search for the book in the Pandas DataFrame
    if matched_book:
        for user_input in matched_book:
            result = all_latest_books[all_latest_books['Book-Title'] == user_input]
            if not result.empty:
                searched_book = {
                    'Book-Title': result['Book-Title'].values[0],
                    'Book-Author': result['Book-Author'].values[0],
                    'Image-URL-M': result['Image-URL-M'].values[0],
                    'Link': result['Link'].values[0]
                }
                data.append(searched_book)

    if matched_author:
        data = []  # List to store book details
        print("matched_author", matched_author)
        # Search for the book in the Pandas DataFrame
        if len(matched_author) == 1:
            books_by_authors = mongo.db.all_latest_books.find({"Book-Author": matched_author[0]}, {"Book-Title": 1})
        else:
            books_by_authors = mongo.db.all_latest_books.find({"Book-Author": {"$in": matched_author}}, {"Book-Title": 1})
        book_titles = [book["Book-Title"] for book in books_by_authors]
        print("Book name against Authors are: ", book_titles)
        for user_input in book_titles:
            result = all_latest_books[all_latest_books['Book-Title'] == user_input]
            if not result.empty:
                searched_book = {
                    'Book-Title': result['Book-Title'].values[0],
                    'Book-Author': result['Book-Author'].values[0],
                    'Image-URL-M': result['Image-URL-M'].values[0],
                    'Link': result['Link'].values[0]
                }
                data.append(searched_book)
        print(data)

    return render_template('admin_view_pdf.html', pdf_data_base64=pdf_data_base64, fav=data)







@app.route('/search_for_book', methods=['POST'])
def search_for_book():
    all_latest_books = pd.DataFrame(list(mongo.db.all_books.find()))
    query = request.form['Book_name']
    matched_book = []
    matched_author = []
    if query:
        # Search for books that match the query
        matched_book = [book for book in all_latest_books['Book-Title'] if query.lower() in book.lower()]
        matched_author = [book for book in all_latest_books['Book-Author'] if query.lower() in book.lower()]
        print("Match Titles", matched_book)
        print("Match Authors", matched_author)
        matched_author = list(set(matched_author))  # drop duplicate

    data = []  # List to store book details
    # Search for the book in the Pandas DataFrame
    if matched_book:
        for user_input in matched_book:
            result = all_latest_books[all_latest_books['Book-Title'] == user_input]
            if not result.empty:
                searched_book = {
                    'Book-Title': result['Book-Title'].values[0],
                    'Book-Author': result['Book-Author'].values[0],
                    'Image-URL-M': result['Image-URL-M'].values[0],
                    'Link': result['Link'].values[0]
                }
                data.append(searched_book)

    if matched_author:
        data = []  # List to store book details
        print("matched_author", matched_author)
        # Search for the book in the Pandas DataFrame
        if len(matched_author) == 1:
            books_by_authors = mongo.db.all_latest_books.find({"Book-Author": matched_author[0]}, {"Book-Title": 1})
        else:
            books_by_authors = mongo.db.all_latest_books.find({"Book-Author": {"$in": matched_author}}, {"Book-Title": 1})
        book_titles = [book["Book-Title"] for book in books_by_authors]
        print("Book name against Authors are: ", book_titles)
        for user_input in book_titles:
            result = all_latest_books[all_latest_books['Book-Title'] == user_input]
            if not result.empty:
                searched_book = {
                    'Book-Title': result['Book-Title'].values[0],
                    'Book-Author': result['Book-Author'].values[0],
                    'Image-URL-M': result['Image-URL-M'].values[0],
                    'Link': result['Link'].values[0]
                }
                data.append(searched_book)
        print(data)

    return render_template('admin.html', num=3, results=data)








@app.route('/process_result', methods=['POST'])
def process_result():
    All_latest_books = pd.DataFrame(list(mongo.db.all_books.find()))
    user_input = request.form['user_input']
    pdf_data_base64 = request.form['pdf_data_base64']

    if not pdf_data_base64:
        return 'No PDF data found'


    print("User input:", user_input)  # Print the user input

    # Render the view again with the PDF data and user input
    # return render_template('view_pdf.html', pdf_data_base64=pdf_data_base64, user_input=user_input, title=title)

    query = user_input
    matched_book = []
    matched_author = []
    if query:
        # Search for books that match the query
        matched_book = [book for book in All_latest_books['Book-Title'] if query.lower() in book.lower()]
        matched_author = [book for book in All_latest_books['Book-Author'] if query.lower() in book.lower()]
        print("Match Titles", matched_book)
        print("Match Authors", matched_author)
        matched_author = list(set(matched_author))  # drop duplicate

    data = []  # List to store book details
    # Search for the book in the Pandas DataFrame
    if matched_book:
        for user_input in matched_book:
            result = All_latest_books[All_latest_books['Book-Title'] == user_input]
            if not result.empty:
                searched_book = {
                    'Book-Title': result['Book-Title'].values[0],
                    'Book-Author': result['Book-Author'].values[0],
                    'Image-URL-M': result['Image-URL-M'].values[0],
                    'Link': result['Link'].values[0]
                }
                data.append(searched_book)

    if matched_author:
        data = []  # List to store book details
        print("matched_author", matched_author)
        # Search for the book in the Pandas DataFrame
        if len(matched_author) == 1:
            books_by_authors = mongo.db.All_latest_books.find({"Book-Author": matched_author[0]}, {"Book-Title": 1})
        else:
            books_by_authors = mongo.db.All_latest_books.find({"Book-Author": {"$in": matched_author}}, {"Book-Title": 1})
        book_titles = [book["Book-Title"] for book in books_by_authors]
        print("Book name against Authors are: ", book_titles)
        for user_input in book_titles:
            result = All_latest_books[All_latest_books['Book-Title'] == user_input]
            if not result.empty:
                searched_book = {
                    'Book-Title': result['Book-Title'].values[0],
                    'Book-Author': result['Book-Author'].values[0],
                    'Image-URL-M': result['Image-URL-M'].values[0],
                    'Link': result['Link'].values[0]
                }
                data.append(searched_book)
        print(data)

    return render_template('admin_view_pdf.html', fav=data, pdf_data_base64=pdf_data_base64)





def update_link(book_title, new_link):
    query = {"Book-Title": book_title}
    new_values = {"$set": {"Link": new_link}}
    print(query, new_values)
    mongo.db.all_books.update_one(query, new_values)


def store_pdf(pdf_path, filename):
    filenames = filename + ".pdf"
    # Open the PDF file in binary mode
    with open(pdf_path, 'rb') as pdf_file:
        fs_id = fs.put(pdf_file, filename=filenames)

    # Get the ObjectId of the stored file
    object_id = fs_id

    new_link = "http://127.0.0.1:5001/open_book/" + str(object_id)
    # Update the link in the database
    update_link(filename, new_link)
    print(f"Link updated to: {new_link}")

    print("PDF stored in collection 'Books' with ObjectId: {object_id}")
    return object_id





@app.route('/upload-file', methods=['POST'])
def upload_pdf():
    try:
        all_latest_book = pd.DataFrame(list(mongo.db.all_books.find()))
        book_title = request.form['book_title']
        pdf_data_base64 = request.form['pdf_data_base64']

        if not pdf_data_base64:
            return jsonify({'error': 'No PDF data found'}), 400

        book_index = all_latest_book[all_latest_book['Book-Title'] == book_title].index[0]
        # Check if the corresponding Link is empty
        # Using pd.isna() to properly check for NaN values.
        if all_latest_book.loc[book_index, 'Link'] is not None and \
                not pd.isna(all_latest_book.loc[book_index, 'Link']) and \
                all_latest_book.loc[book_index, 'Link'] != "":
            return jsonify({'message': 'Book PDF already exists', 'link': all_latest_book.loc[book_index, 'Link']}), 200

        pdf_name = f"{book_title}.pdf"

        # Decode the base64 data and save it to a PDF file
        pdf_path = f"{book_title.replace(' ', '_')}.pdf"
        with open(pdf_path, 'wb') as pdf_file:
            pdf_file.write(base64.b64decode(pdf_data_base64))

        store_pdf(pdf_path, book_title)

        # Return a success message
        return jsonify({'message': 'PDF file saved successfully'}), 200

    except Exception as e:
        print(f"An error occurred: {e}")
        return jsonify({'error': str(e)}), 500







fs_images = GridFS(mongo.db, collection="Images")

def generate_image_url(image_id):
    return url_for('get_image', image_id=image_id, _external=True)

@app.route('/add_book', methods=['POST'])
def add_book():
    title = request.form['title']
    author = request.form['author']
    publication_year = request.form['publication_year']
    category = request.form['category']
    comment = request.form['comment']
    image_file = request.files['image']  # Change to request.files to handle file uploads

    # Check if a book with the same title already exists
    existing_book = mongo.db.all_books.find_one({'Book-Title': title})
    if existing_book:
        return '<div style="color: red; font-size: 50px; text-align: center; position: fixed; top: 50%; left: 50%; transform: translate(-50%, -50%);">Error: Book already exists in the database</div>'



    # Save the image to GridFS
    image_id = fs_images.put(image_file.read(), filename=image_file.filename, content_type=image_file.content_type)

    # Generate the image URL
    image_url = generate_image_url(image_id)

    # Create the book document
    book = {
        'Book-Title': title,
        'Book-Author': author,
        'Num-Rating': 0,
        'Avg-Rating': 0,
        'Year-Of-Publication': publication_year,
        'Image-URL-M': image_url,
        'Link': None,
        'Category': category,
        'Summary': comment
    }

    # Insert the book document into the allbooks collection in MongoDB
    mongo.db.all_books.insert_one(book)
    return 'Book added successfully'


from io import BytesIO


@app.route('/image/<image_id>')
def get_image(image_id):
    try:
        image = fs_images.get(ObjectId(image_id))
        return send_file(BytesIO(image.read()), mimetype=image.content_type)
    except Exception as e:
        return str(e), 404


@app.route('/delete_book_pdf', methods=['POST'])
def delete_pdf():
    try:
        # Extract book title from the form data
        book_title = request.form['book_title']

        # Find the book by title in the database
        book = mongo.db.all_books.find_one({"Book-Title": book_title})

        if not book:
            return jsonify({'error': 'Book not found'}), 404

        # Extract the ObjectId from the Link field
        link = book.get('Link')
        print(link)
        if pd.isna(link) or link == "" or link is None:
            return jsonify({'error': 'No PDF link found for this book'}), 400

        # Convert link to string before splitting
        link_str = str(link)
        object_id_str = link_str.split('/')[-1]
        object_id = ObjectId(object_id_str)

        # Remove the file from GridFS
        fs.delete(object_id)

        # Update the book's Link to null
        query = {"Book-Title": book_title}
        new_values = {"$set": {"Link": None}}
        mongo.db.all_books.update_one(query, new_values)

        return jsonify({'message': 'PDF file deleted successfully'}), 200

    except Exception as e:
        print(f"An error occurred: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/delete_book', methods=['POST'])
def delete_book():
    book_name = request.form['book_title']

    book = mongo.db.all_books.find_one({'Book-Title': book_name})
    if not book:
        return 'Book not found'

    link = book.get('Link')

    print(link)
    if link:
        return jsonify({'error': 'Kindly delete the Book PDF file first.'}), 40

    # Extract the image URL from the book document
    image_url = book.get('Image-URL-M')
    if not image_url:
        return 'Image URL not found for the book'

    # Extract the image ID from the image URL
    image_id = ObjectId(image_url.split('/')[-1])

    # Delete the image files from GridFS
    fs_images.delete(image_id)

    mongo.db.Rating.delete_many({"Book-Name": book_name})

    mongo.db.users.update_many(
        {
            "$or": [
                {"Recommendations": {"$in": [book_name]}},
                {"Fav_Recommendations": {"$in": [book_name]}},
                {"favorites": {"$in": [book_name]}}
            ]
        },
        {
            "$pull": {
                "Recommendations": book_name,
                "Fav_Recommendations": book_name,
                "favorites": book_name
            }
        }
    )

    mongo.db.all_books.delete_one({"Book-Title": book_name})

    return 'Book and associated image files deleted successfully'






@app.route('/home')
def popular_books():
    # Popular Books
    all_book = pd.DataFrame(list(mongo.db.all_books.find()))
    all_book['Avg-Rating'] = all_book['Avg-Rating'] / 2

    recent_books = all_book.iloc[::-1].head(100)


    filtered_books = all_book[all_book['Num-Rating'] >= 5]
    sorted_popular_df = filtered_books.sort_values(by='Avg-Rating', ascending=False)
    sorted_popular_df = sorted_popular_df.head(100)
    # Most Rated Books
    most_rated = all_book[all_book['Num-Rating'] >= 0]
    sorted_popular_dfs = most_rated.sort_values(by='Num-Rating', ascending=False)
    sorted_popular_dfs = sorted_popular_dfs.head(100)

    filtered_book = all_book[all_book['Num-Rating'] > 3]
    sort = filtered_book.sort_values(by='Avg-Rating', ascending=False)
    book_fiction = sort[sort['Category'] == 'Fiction'].head(100)

    filters = all_book[all_book['Num-Rating'] > 3]
    sorts = filters.sort_values(by='Avg-Rating', ascending=False)
    book_non_fiction = sorts[sorts['Category'] == 'Others'].head(100)

    # Pass variables to the template
    popular_data = {
        'book_name': sorted_popular_df['Book-Title'].to_list(),
        'author': sorted_popular_df['Book-Author'].to_list(),
        'image': sorted_popular_df['Image-URL-M'].to_list(),
        'votes': sorted_popular_df['Num-Rating'].to_list(),
        'rating': sorted_popular_df['Avg-Rating'].to_list(),
        'most_viewed_book_name': sorted_popular_dfs['Book-Title'].to_list(),
        'most_viewed_author': sorted_popular_dfs['Book-Author'].to_list(),
        'most_viewed_image': sorted_popular_dfs['Image-URL-M'].to_list(),
        'most_viewed_votes': sorted_popular_dfs['Num-Rating'].to_list(),
        'most_viewed_rating': sorted_popular_dfs['Avg-Rating'].to_list(),
        'book_fiction_name': book_fiction['Book-Title'].to_list(),
        'book_fiction_author': book_fiction['Book-Author'].to_list(),
        'book_fiction_image': book_fiction['Image-URL-M'].to_list(),
        'book_fiction_votes': book_fiction['Num-Rating'].to_list(),
        'book_fiction_rating': book_fiction['Avg-Rating'].to_list(),
        'book_non_fiction_name': book_non_fiction['Book-Title'].to_list(),
        'book_non_fiction_author': book_non_fiction['Book-Author'].to_list(),
        'book_non_fiction_image': book_non_fiction['Image-URL-M'].to_list(),
        'book_non_fiction_votes': book_non_fiction['Num-Rating'].to_list(),
        'book_non_fiction_rating': book_non_fiction['Avg-Rating'].to_list(),
        'recent_book_name': recent_books['Book-Title'].to_list(),
        'recent_author': recent_books['Book-Author'].to_list(),
        'recent_image': recent_books['Image-URL-M'].to_list(),
        'recent_votes': recent_books['Num-Rating'].to_list(),
        'recent_rating': recent_books['Avg-Rating'].to_list()
    }

    return render_template('index.html', **popular_data)


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
        return send_file(temp_path, as_attachment=False)
    else:
        return 'Book not found', 404


@app.route('/add_to_favorites', methods=['POST'])
def add_to_favorites():
    if 'username' in session:
        username = session['username']
        book_name = request.form.get('book_name')
        print(book_name)
        users = mongo.db.users
        user = users.find_one({'username': username, 'favorites': book_name})
        if user:
            # Book already in Fav
            # flash('Thank you for rating!', 'success')
            return jsonify({'success': False, 'message': f"'{book_name}' is already in your favorites."}), 200

        # Add the book to favorites for the current user in the database
        print(book_name)
        print(f"Book '{book_name}' added to favorites for user '{username}'")  # Print message to console
        users_fav_recommendation(book_name)
        users.update_one({'username': username}, {'$addToSet': {'favorites': book_name}})
        return jsonify({'success': False, 'message': f"'{book_name}' added to favorites."}), 200
    else:
        return redirect(url_for('login'))


@app.route('/Favorite')
def display_favorites():
    if 'username' in session:
        fav = []
        username = session['username']
        user_data = mongo.db.users.find_one({'username': username})
        all_books_ = pd.DataFrame(list(mongo.db.all_books.find()))
        if user_data and 'favorites' in user_data:
            favorite_books = user_data['favorites'][::-1]
            for favorite_book in favorite_books:
                print(favorite_book)
                result = all_books_[all_books_['Book-Title'] == favorite_book]
                # Check if the book is found
                if not result.empty:
                    # Append details of the searched book to the list
                    fav.append({
                        'Book-Title': result['Book-Title'].iloc[0],
                        'Book-Author': result['Book-Author'].iloc[0],
                        'Image-URL-M': result['Image-URL-M'].iloc[0],
                        'votes': result['Num-Rating'].iloc[0],
                        'rating': result['Avg-Rating'].iloc[0] / 2
                    })
                    print(fav)

        else:
            return render_template('Favorite.html', favorite_books=[])

        return render_template('Favorite.html', fav=fav)
    else:
        return redirect(url_for('login'))


@app.route('/Recommend')
def display_recommend():
    if 'username' in session:
        rec = []
        fav_rec = []
        username = session['username']
        user_data = mongo.db.users.find_one({'username': username})
        all_books_ = pd.DataFrame(list(mongo.db.all_books.find()))
        if user_data and 'Recommendations' in user_data:
            Recommend_books = user_data['Recommendations'][::-1]
            for recommend_book in Recommend_books:
                print(recommend_book)
                result = all_books_[all_books_['Book-Title'] == recommend_book]
                # Check if the book is found
                if not result.empty:
                    # Append details of the searched book to the list
                    rec.append({
                        'Book-Title': result['Book-Title'].iloc[0],
                        'Book-Author': result['Book-Author'].iloc[0],
                        'Image-URL-M': result['Image-URL-M'].iloc[0],
                        'votes': result['Num-Rating'].iloc[0],
                        'rating': result['Avg-Rating'].iloc[0] / 2
                    })
                    print(rec)
        if user_data and 'Fav_Recommendations' in user_data:
            Recommend_books = user_data['Fav_Recommendations'][::-1]
            for recommend_book in Recommend_books:
                print(recommend_book)
                result = all_books_[all_books_['Book-Title'] == recommend_book]
                # Check if the book is found
                if not result.empty:
                    # Append details of the searched book to the list
                    fav_rec.append({
                        'Book-Title': result['Book-Title'].iloc[0],
                        'Book-Author': result['Book-Author'].iloc[0],
                        'Image-URL-M': result['Image-URL-M'].iloc[0],
                        'votes': result['Num-Rating'].iloc[0],
                        'rating': result['Avg-Rating'].iloc[0] / 2
                    })
                    print(fav_rec)

        return render_template('recommend.html', rec=rec, fav_rec=fav_rec)
    else:
        return redirect(url_for('login'))


@app.route('/remove_favorite', methods=['POST'])
def remove_favorite():
    if 'username' in session:
        username = session['username']
        book_title = request.form.get('book_title')  # Assuming the book title is sent via form
        # Remove the book from the user's favorites in the database
        mongo.db.users.update_one({'username': username}, {'$pull': {'favorites': book_title}})
        return redirect(url_for('display_favorites'))
    else:
        return redirect(url_for('login'))


@app.route('/remove_recommend', methods=['POST'])
def remove_recommend():
    if 'username' in session:
        username = session['username']
        book_title = request.form.get('book_title')  # Assuming the book title is sent via form
        # Remove the book from the user's favorites in the database
        mongo.db.users.update_one({'username': username}, {'$pull': {'Recommendations': book_title}})
        return redirect(url_for('display_recommend'))
    else:
        return redirect(url_for('login'))


@app.route('/remove_fav_recommend', methods=['POST'])
def remove_fav_recommend():
    if 'username' in session:
        username = session['username']
        book_title = request.form.get('book_title')  # Assuming the book title is sent via form
        # Remove the book from the user's favorites in the database
        mongo.db.users.update_one({'username': username}, {'$pull': {'Fav_Recommendations': book_title}})
        return redirect(url_for('display_recommend'))
    else:
        return redirect(url_for('login'))


@app.route('/submit-rating', methods=['POST'])
def submit_rating():
    if request.method == 'POST':
        rating = float(request.form['rating'])
        comment = request.form['comment']
        book_name = request.form['book_name']
        username = session['username']

        # Check if the user has already rated this book
        existing_rating = mongo.db.Rating.find_one({'Username': username, 'Book-Name': book_name})

        # if user forcefully enter rating again
        if existing_rating:
            all_book = pd.DataFrame(list(mongo.db.all_books.find()))
            existing_book_Avg_rating = all_book.loc[all_book['Book-Title'] == existing_rating['Book-Name'], 'Avg-Rating'].values[0] / 2
            existing_book_Num_rating = all_book.loc[all_book['Book-Title'] == existing_rating['Book-Name'], 'Num-Rating'].values[0]

            # Avg-Rating_new = (Avg-Rating_old×(Num-Rating−1))+(2×rating) / Num-Rating
            # Avg-Rating_old = ((Avg-Rating_new×Num-Rating)+(2×existing_rating['Rating']))/(Num-Rating−1)
            old_avg_rating = ((existing_book_Avg_rating * existing_book_Num_rating) - (existing_rating['Rating'])) / (existing_book_Num_rating - 1)
            print(old_avg_rating)
            updated_avg_rating = ((old_avg_rating*(existing_book_Num_rating-1))+rating) / existing_book_Num_rating

            mongo.db.Rating.update_one(
                {'_id': existing_rating['_id']},
                {'$set': {'Rating': rating, 'Comment': comment}}
            )
            mongo.db.all_books.update_one(
                {'Book-Title': existing_rating['Book-Name']},
                {'$set': {'Avg-Rating': updated_avg_rating * 2}}
            )
            print("vbn")

        else:
            # Update Num-Rating by 1 and calculate new Avg-Rating
            # Avg-Rating = (Avg-Rating×(Num-Rating−1))−(2×rating) / Num-Rating
            mongo.db.all_books.update_one(
                {'Book-Title': book_name},
                [
                    {'$set': {'Num-Rating': {'$add': ['$Num-Rating', 1]}}},
                    {'$set': {
                        'Avg-Rating': {
                            '$divide': [
                                {'$add': [
                                    {'$multiply': ['$Avg-Rating', {'$subtract': ['$Num-Rating', 1]}]},
                                    {'$multiply': [rating, 2]}
                                ]},
                                '$Num-Rating'
                            ]
                        }
                    }}
                ]
            )

            # Insert rating into Rating collection
            mongo.db.Rating.insert_one({
                'Username': username,
                'Book-Name': book_name,
                'Rating': rating,
                'Comment': comment
            })

        # flash('Thank you for rating!', 'success')
        return redirect(url_for('rating_page', book_name=book_name))


@app.route('/rating-page', methods=['GET', 'POST'])
def rating_page():
    if request.method == 'POST':
        if 'book_name' in request.form:
            book_name = request.form['book_name']
        else:
            book_name = []
    else:
        book_name = request.args.get('book_name', '')

    username = session['username']
    user_already_rated = False

    ratings_cursor = mongo.db.Rating.find({'Book-Name': book_name})
    ratings_df = pd.DataFrame(list(ratings_cursor))

    if not ratings_df.empty:
        print("yes")
        ratings_dfs = ratings_df.drop(columns=['Book-Name', '_id'])
        ratings_df = ratings_dfs.to_dict('records')
        print(ratings_df)
        for row in ratings_df:
            print(row['Username'])
            if row['Username'] == username:
                user_already_rated = True
                break
    else:
        print("no")
        ratings_df = []

    all_book = pd.DataFrame(list(mongo.db.all_books.find()))
    all_book['Avg-Rating'] = all_book['Avg-Rating'] / 2
    book_details = all_book[all_book['Book-Title'] == book_name].reset_index(drop=True)
    data = {
        "Book-Title": book_details.at[0, 'Book-Title'],
        "Book-Author": book_details.at[0, 'Book-Author'],
        "Num-Rating": book_details.at[0, 'Num-Rating'],
        "Avg-Rating": round(book_details.at[0, 'Avg-Rating'], 3),
        "Year-Of-Publication": book_details.at[0, 'Year-Of-Publication'],
        "Image-URL-M": book_details.at[0, 'Image-URL-M'],
        "Link": book_details.at[0, 'Link'],
        "Category": book_details.at[0, 'Category'],
        "Summary": book_details.at[0, 'Summary']
    }
    print(data)
    print(data['Book-Title'])
    return render_template('book_detail.html', ratings=ratings_df, data=data, user_already_rated=user_already_rated)


@app.route('/delete-rating', methods=['POST'])
def delete_comment_rating():
    book_name = request.form['book_name']
    username = session['username']
    existing_rating = mongo.db.Rating.find_one({'Username': username, 'Book-Name': book_name})
    if existing_rating:
        all_book = pd.DataFrame(list(mongo.db.all_books.find()))
        existing_book_Avg_rating = all_book.loc[all_book['Book-Title'] == existing_rating['Book-Name'], 'Avg-Rating'].values[0] / 2
        existing_book_Num_rating = all_book.loc[all_book['Book-Title'] == existing_rating['Book-Name'], 'Num-Rating'].values[0]
        old_avg_rating = ((existing_book_Avg_rating * existing_book_Num_rating) - (existing_rating['Rating'])) / (existing_book_Num_rating - 1)
        mongo.db.all_books.update_one(
            {'Book-Title': existing_rating['Book-Name']},
            {
                '$set': {'Avg-Rating': old_avg_rating * 2},
                '$inc': {'Num-Rating': -1}
            }
        )
        mongo.db.Rating.delete_one({'_id': existing_rating['_id']})
    return redirect(url_for('rating_page', book_name=book_name))


@app.route('/check-forget-email', methods=['POST'])
def check_email_and_send_otp():
    data = request.json
    email = data.get('email')

    # Check if the email already exists in MongoDB
    users = mongo.db.users
    existing_email = users.find_one({'email': email})

    if existing_email:
        # Email already exists, return response indicating not valid
        try:
            send_otp_email(email)
            return jsonify({'valid': True, 'messageSent': True})
        except Exception as e:
            print(e)
            return jsonify({'valid': True, 'messageSent': False})
    else:
        return jsonify({'valid': False, 'message': 'Email not exists in the database'})


def generate_otp():
    return ''.join(random.choices(string.digits, k=6))


def send_otp_email(to_email):
    your_email = os.environ.get('EMAIL_ADDRESS')
    your_password = os.environ.get('EMAIL_PASSWORD')

    print("Email Address:", os.environ.get('EMAIL_ADDRESS'))
    print("Email Password:", os.environ.get('EMAIL_PASSWORD'))

    # Setting up the SMTP server
    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.login(your_email, your_password)

    # Creating the email content
    msg = MIMEMultipart()
    msg['From'] = your_email
    msg['To'] = to_email
    msg['Subject'] = "OTP Verification"
    otp = generate_otp()
    body = f"Your OTP is: {otp}"
    msg.attach(MIMEText(body, 'plain'))
    # Sending the email
    server.send_message(msg)
    del msg  # Delete the message object after sending
    server.quit()

    global otp_code
    otp_code = otp
    print("Global OTP is: ", otp_code)
    # Start a new thread to run expire_otp() concurrently
    expiry_thread = threading.Thread(target=expire_otp)
    expiry_thread.start()


def expire_otp():
    """Function to expire OTP after 30 seconds."""
    global otp_code
    # Wait for 30 seconds
    time.sleep(10)
    # Reset the OTP after expiration
    otp_code = '1'
    print("OTP expired: ", otp_code)


@app.route('/forget', methods=['GET', 'POST'])
def forget_password():
    if request.method == 'POST':
        print("d")

    else:
        # This part of the code will render the forget password form when the page is initially loaded
        return render_template('forget_password.html')


@app.route('/reset', methods=['GET'])
def reset():
    email = request.args.get('email')
    print(email)
    return render_template('reset_password.html', email=email)


@app.route('/verify-otp', methods=['POST'])
def verify_otp():
    global otp_code
    email = request.form.get('emails')
    submitted_otp = request.form.get('otp')
    print(submitted_otp, "=", otp_code)
    if submitted_otp == otp_code and otp_code != '1':
        return render_template('change_password.html', email=email)
    else:
        error_message = "Invalid or Expire OTP, Please try again!"
        print(email)
        return render_template('reset_password.html', error=error_message, email=email)


@app.route('/change-password', methods=['POST'])
def change_password():
    email = request.form['email']
    password = request.form['password']
    confirm_password = request.form['confirm_password']

    if password != confirm_password:
        error_message = 'Password and Confirm Password do not match.'
        return render_template('change_password.html', email=email, error_message=error_message)
    else:
        # Update password in the database for the user with the provided email
        users = mongo.db.users
        users.update_one({'email': email}, {'$set': {'password': generate_password_hash(password)}})
        user = users.find_one({'email': email})
        username = user.get('username', 'User')

        success_message = f"{username} Password successfully changed."
        return render_template('login.html', success_message=success_message)


if __name__ == '__main__':
    app.run(debug=True, port=5001)
 #, use_reloader=False
