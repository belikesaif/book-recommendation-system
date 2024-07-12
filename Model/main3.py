import pandas as pd
import numpy as np
import pickle

# Load the dataset
data = pd.read_csv('Data/Preprocessed_data.csv', low_memory=False)


def preprocess_data(dfa):
    df = dfa.copy()
    # Handle missing values, removing rows where 'Book-Title' or 'Book-Author' are NaN or only whitespace upfront
    df.dropna(subset=['Book-Title', 'Book-Author'], inplace=True)
    df = df[df['Book-Title'].str.strip() != '']
    df = df[df['Book-Author'].str.strip() != '']
    # Adjust book titles: remove spaces at the ends and capitalize words
    df['Book-Title'] = df['Book-Title'].str.strip().str.title()
    # Remove rows with special characters in 'Book-Title' and 'Book-Author'
    df = df[~df['Book-Title'].str.contains(r"[^\w\s]", regex=True)]
    df = df[~df['Book-Author'].str.contains(r"[^\w\s]", regex=True)]
    # Fill missing 'Image-URL-M' values
    df['Image-URL-M'].fillna('Not Available', inplace=True)
    # Handle 'Year-Of-Publication' anomalies and convert to numeric
    df['Year-Of-Publication'].replace(['DK Publishing Inc', 'Gallimard'], np.nan, inplace=True)
    df['Year-Of-Publication'] = pd.to_numeric(df['Year-Of-Publication'], errors='coerce')
    # Convert 'Age' to numeric, handling non-numeric with coercion, and fill missing values with the mean
    df['Age'] = pd.to_numeric(df['Age'], errors='coerce').fillna(df['Age'].mean())
    # Remove titles that contain a specific pattern of numbers (e.g., ISBNs or years)
    df = df[~df['Book-Title'].str.contains(r'^\d+|\b\d{5}\b', regex=True)]
    # Clean 'Category' from special characters and replace specific unwanted values
    df['Category'] = df['Category'].str.replace(r"[^\w\s]", '', regex=True).replace('9', 'Others')
    df = df[~df['Category'].str.contains(r'\d', regex=True)]
    # Handle missing values, removing rows where 'Book-Title' or 'Book-Author' are NaN or only whitespace upfront
    df.dropna(subset=['Book-Title', 'Book-Author'], inplace=True)
    return df

# Preprocess the data
processed_data = preprocess_data(data)

# Group by Book-Title and User-ID, then filter groups with at least 3 ratings
users_with_min_3_ratings = processed_data.groupby(['Book-Title', 'User-ID']).filter(lambda x: len(x) >= 3)
# Calculate the average age for each book title
average_age_per_book = users_with_min_3_ratings.groupby('Book-Title')['Age'].mean().reset_index(name='Avg-Age')
# Avg Rating
book_and_rating = processed_data.groupby('Book-Title').agg({'Book-Rating': ['count', 'mean']})
book_and_rating.columns = ['Num-Rating', 'Avg-Rating']
# Choose All Books
popular_rating = book_and_rating.reset_index()
popular_rating = popular_rating[(popular_rating['Num-Rating'] > -1)].sort_values('Avg-Rating', ascending=False)
# Join the average age with the popular_rating DataFrame
popular_rating_with_age = popular_rating.merge(average_age_per_book, on='Book-Title', how='left')
# Join with the processed_data to get additional information for the website
all_books = popular_rating_with_age.merge(processed_data.drop_duplicates('Book-Title'), on='Book-Title')
#make all books have Avg rating out of 5 instead of 10
all_books['Avg-Rating'] /= 2

# Select the specified columns
all_books = all_books[
    ['Book-Title', 'Book-Author', 'Num-Rating', 'Avg-Rating', 'Year-Of-Publication', 'Image-URL-M', 'Link', 'Category', 'Avg-Age']]


print("Number of unique categories in dataset:", data['Category'].nunique())
unique_categories_count = all_books['Category'].nunique()
print("Total number of unique categories that we include:", unique_categories_count)

category_counts = all_books['Category'].value_counts()
print("Count of books in each category:")
print(category_counts)

for category, count in category_counts.items():
   print(f"{category}: {count}")

def print_books_from_df(df):
    for title in df['Book-Title']:
        print(title)

# Call the function with the all_books DataFrame
print_books_from_df(all_books)
print(all_books.shape)
print(all_books.size)

#pickle.dump(all_books, open('artifacts/all_books.pkl', 'wb'))
