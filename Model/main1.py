import numpy as np
import pandas as pd
import pickle
import requests

# Load the dataset
data = pd.read_csv('Data/Preprocessed_data.csv', low_memory=False)


def fetch_image_size(url):
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
        response = requests.head(url, headers=headers, timeout=30)  # Include User-Agent header
        print("Response Headers:", response.headers)  # Print response headers for debugging
        if response.status_code == 200:  # Check if the request was successful
            content_size = int(response.headers.get('Content-Length', 0))  # Get content size from the 'Content-Length' header
            return content_size
        else:
            print(f"Error: Received status code {response.status_code} for URL {url}")
            return None
    except Exception as e:
        print(f"Error fetching URL {url}: {e}")
        return None


def preprocess_data(dfa):
    df = dfa.copy()
    df = df[df['Book-Title'].str.strip() != '']
    df = df[df['Book-Author'].str.strip() != '']
    # Adjust book titles: remove spaces at the ends and capitalize words
    df['Book-Title'] = df['Book-Title'].str.strip().str.title()
    # Remove rows with special characters in 'Book-Title' and 'Book-Author'
    df = df[~df['Book-Title'].str.contains(r"[^\w\s]", regex=True)]
    df = df[~df['Book-Author'].str.contains(r"[^\w\s]", regex=True)]
    # Convert 'Age' to numeric, handling non-numeric with coercion, and fill missing values with the mean
    df['Age'] = pd.to_numeric(df['Age'], errors='coerce').fillna(df['Age'].mean())
    # Remove rows that titles contain a specific pattern of numbers (e.g., ISBNs or years)
    df = df[~df['Book-Title'].str.contains(r'^\d+|\b\d{5}\b', regex=True)]
    # Clean 'Category' from special characters and replace specific unwanted values
    df['Category'] = df['Category'].str.replace(r"[^\w\s]", '', regex=True).replace('9', 'Others')
    df['Summary'] = df['Summary'].str.replace(r"[^\w\s]", '', regex=True).replace('9', 'Not Available')
    df['Category'] = [' '.join(word for word in category.split() if word) for category in df['Category']]
    df['Category'] = df['Category'].str.lower()
    df['Category'] = df['Category'].apply(lambda x: ' '.join(word.capitalize() for word in x.split()))
    # remove category rows that contain numbers
    print(df.shape)
    df = df[~df['Category'].str.contains(r'\d', regex=True)]
    # Fill missing 'Image-URL-M' values
    df['Image-URL-M'].fillna('Not Available', inplace=True)
    # Handle 'Year-Of-Publication' anomalies and convert to numeric
    df['Year-Of-Publication'].replace(['DK Publishing Inc', 'Gallimard'], np.nan, inplace=True)
    df['Year-Of-Publication'] = pd.to_numeric(df['Year-Of-Publication'], errors='coerce')
    return df

# Preprocess the data
processed_data = preprocess_data(data)
# Avg Rating

book_and_rating = processed_data.groupby('Book-Title').agg({'Book-Rating': ['count', 'mean']})
book_and_rating.columns = ['Num-Rating', 'Avg-Rating']
# Choose All Books
popular_rating = book_and_rating.reset_index()
# Calculate the average age for each book title
average_age_per_book = processed_data.groupby('Book-Title')['Age'].mean().reset_index(name='Avg-User-Age')

# Join the average age with the popular_rating DataFrame
popular_rating_with_age = popular_rating.merge(average_age_per_book, on='Book-Title', how='left')
# Join with the processed_data to get additional information for the website
all_books = popular_rating_with_age.merge(processed_data.drop_duplicates('Book-Title'), on='Book-Title', how='left')

# Select the specified columns
all_books = all_books[
    ['Book-Title', 'Book-Author', 'Num-Rating', 'Avg-Rating', 'Year-Of-Publication', 'Image-URL-M', 'Link', 'Category', 'Avg-User-Age', 'Summary']]

print(all_books.shape)

#remove dublicates if Book-Title = Book-Author will dublicates
all_books.drop_duplicates(subset=['Book-Title', 'Book-Author'], keep='last', inplace=True)

print(all_books.shape)
rows_to_remove = []  # List to store indices of rows to remove
# Iterate over each row in the DataFrame
i = 0
for index, row in all_books.iterrows():
    i = i + 1
    print(i)
    url = row['Image-URL-M']
    print(url)
    image_size = fetch_image_size(url)
    if image_size is not None and image_size < 50:
        # If image size is less than 50 bytes, add the index to the list of rows to remove
        rows_to_remove.append(index)

# Remove rows with indices stored in the rows_to_remove list
all_books = all_books.drop(rows_to_remove)
print(all_books.shape)
book_names = all_books[['Book-Title']]

pickle.dump(all_books, open('artifacts/all_books.pkl', 'wb'))
pickle.dump(book_names, open('artifacts/book_names.pkl', 'wb'))


print("After Preprocessing Books were:", all_books.shape)


category_counts = all_books['Category'].value_counts()
print("Total Number of Category: ", category_counts.shape[0])
print("Count of books in each category:")
print(category_counts)








# Preprocess the data
processed = preprocess_data(data)
print(processed.shape)
processed_data = pd.merge(processed, all_books,
                          on=['Image-URL-M', 'Book-Title', 'Book-Author', 'Link', 'Year-Of-Publication'], how='inner')
print(processed_data.shape)

books = all_books

#Colleborating Base Filtering based on Users
#KNN Approach
#similar item

x = processed_data.groupby('User-ID').count()['Book-Rating'] > 50 #(<50 ...57030 users, 100 ...)
print("x", x.size)
users = x[x].index #Boolein indexing
filtered_rating = processed_data[processed_data['User-ID'].isin(users)]
y = filtered_rating.groupby('Book-Title').count()['Book-Rating'] >= 0 #(0 ...51015 books)
print("y", y.size)
famous_books = y[y].index
#famous_books = filtered_rating['Book-Title'].unique()
final_ratings = filtered_rating[filtered_rating['Book-Title'].isin(famous_books)]
book_pivot = final_ratings.pivot_table(index='Book-Title', columns='User-ID', values='Book-Rating')
book_pivot.fillna(0,inplace=True) #replace NaN with 0
from sklearn.metrics.pairwise import cosine_similarity
similarity_scores = cosine_similarity(book_pivot)

print(book_pivot.shape)


def recommend(book_name):
    # index fetch
    index = np.where(book_pivot.index == book_name)[0][0]
    similar_items = sorted(list(enumerate(similarity_scores[index])), key=lambda x: x[1], reverse=True)[1:5]
    searched_book = []
    temp_df = books[books['Book-Title'] == book_name]
    searched_book.extend(temp_df.drop_duplicates('Book-Title')['Book-Title'].to_list())

    recommended_books = []
    for i in similar_items:
        temp_df = books[books['Book-Title'] == book_pivot.index[i[0]]]
        recommended_books.extend(temp_df.drop_duplicates('Book-Title')['Book-Title'].to_list())

    return searched_book + recommended_books


books.drop_duplicates('Book-Title')
book_names = book_pivot.index.values

print(book_names)
print("waiting")



# Drop duplicates from the DataFrame if needed
recommendations_df = pd.DataFrame([recommend(book_name) for book_name in book_names],
                                   columns=['Searched_Book', 'Recommended_Book1',
                                            'Recommended_Book2', 'Recommended_Book3', 'Recommended_Book4'])
print(recommendations_df.size)
recommendations_df.drop_duplicates(inplace=True)
print(recommendations_df.size)



def recommend(user_input):
    data = []  # List to store book details

    # Search for the book in the Pandas DataFrame
    result = recommendations_df[recommendations_df['Searched_Book'] == user_input]

    # Check if the book is found
    if not result.empty:
        # Append details of the searched book to the list
        searched_book = {
            "Book Title": result['Searched_Book'].values[0],
        }
        data.append({"Searched Book": searched_book})

        # Append details of recommended books to the list
        recommended_books = []
        for i in range(1, 5):
            recommended_book_key = f"Recommended_Book{i}"

            recommended_book = {
                "Book Title": result[recommended_book_key].values[0],
            }
            recommended_books.append(recommended_book)

        data.append({"Recommended Books": recommended_books})
    else:
        print(f"Book '{user_input}' not found in recommendations_df.")

    return data

print(recommend("Touching Evil"))



# Save the popular_rating DataFrame

recommendations_df.to_pickle('artifacts/recommendations_df.pkl')
print("Pickled final_popular_rating successfully")
