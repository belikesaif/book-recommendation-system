
import pandas as pd
import pickle

all_book = pd.read_csv('Data/DB_all_books.csv', low_memory=False)


all_books = all_book.copy()
all_books['Category'] = [' '.join(word for word in category.split() if word) for category in all_books['Category']]
all_books['Category'] = all_books['Category'].str.lower()
all_books['Category'] = all_books['Category'].apply(lambda x: ' '.join(word.capitalize() for word in x.split()))
# all_books.drop_duplicates(subset='Category', inplace=True)  # Drop duplicates based on modified 'Category' column

print(all_books['Category'].size)
# pickle.dump(all_books, open('artifacts/all_books.pkl', 'wb'))
