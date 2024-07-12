from pymongo import MongoClient
import pickle

# Connect to MongoDB
client = MongoClient("mongodb+srv://zain:mirha456@cluster0.hog4kvb.mongodb.net/")
db = client["myDatabase"]

# Load pickle files
all_books = pickle.load(open('Model/artifacts/all_books.pkl', 'rb'))
recommendations_df = pickle.load(open('Model/artifacts/recommendations_df.pkl', 'rb'))
db["all_books"].insert_many(all_books.to_dict(orient="records"))
db["recommendations_df"].insert_many(recommendations_df.to_dict(orient="records"))

# Close MongoDB connection
client.close()
