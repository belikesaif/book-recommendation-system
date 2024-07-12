import numpy as np
import pandas as pd
import requests
import time

# Load the dataset
try:
    data = pd.read_csv('Data/q.csv', low_memory=False)
except Exception as e:
    print("Error loading dataset:", e)
    raise

def check_image_size(url):
    if url != 'Not Available':
        retries = 3  # Number of retries
        for attempt in range(retries):
            try:
                response = requests.get(url, timeout=30)  # Increase timeout to 30 seconds
                content_size = len(response.content)
                return url, content_size
            except Exception as e:
                print(f"Error fetching URL {url}: {e}")
                if attempt < retries - 1:
                    print(f"Retrying ({attempt + 1}/{retries})...")
                    time.sleep(2)  # Wait for a few seconds before retrying
                else:
                    return url, None
    else:
        return url, None  # Treat 'Not Available' URLs as None

def preprocess_data(dfa, sample_size=100):
    df = dfa.sample(sample_size)  # Sample a subset of rows

    deleted_urls = []  # List to store deleted URLs

    image_sizes = df['Image-URL-M'].apply(check_image_size)
    for url, size in image_sizes:
        if size is not None:
            print(f"Image size for URL {url}: {size} bytes")
        else:
            deleted_urls.append(url)  # Add deleted URL to the list

    valid_urls = [url for url, size in image_sizes if size is not None and size >= 50]
    df = df[df['Image-URL-M'].isin(valid_urls)]

    print("Preprocessing completed.")
    return df, deleted_urls  # Return the list of deleted URLs

# Preprocess the data
try:
    processed_data, deleted_urls = preprocess_data(data, sample_size=1000)  # Adjust sample size as needed
except Exception as e:
    print("Error preprocessing data:", e)
    raise

# Select the specified columns
all_books = processed_data[
    ['Book-Title', 'Book-Author', 'Num-Rating', 'Avg-Rating', 'Year-Of-Publication', 'Image-URL-M', 'Link', 'Category', 'Avg-Age']]

# Print the first few books for debugging
print("First few books:")
print(all_books.head())

# Print the deleted URLs
print("Deleted URLs:")
for url in deleted_urls:
    print(url)
