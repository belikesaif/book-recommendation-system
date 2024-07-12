import pymongo

client = pymongo.MongoClient("mongodb://localhost:27017/myDatabase")
mydatabase = client["myDatabase"]
popular_df = mydatabase["popular_df"]
recommendations_df = mydatabase["recommendations_df"]

def update_link(book_title, new_link):
    query = {"Book-Title": book_title}
    new_values = {"$set": {"Link": new_link}}
    popular_df.update_one(query, new_values)

    query = {"Searched_Book": book_title}
    new_values = {"$set": {"Searched_Link": new_link}}
    recommendations_df.update_one(query, new_values)

    query = {"Recommended_Book1": book_title}
    new_values = {"$set": {"Recommended_Link1": new_link}}
    recommendations_df.update_one(query, new_values)

    query = {"Recommended_Book2": book_title}
    new_values = {"$set": {"Recommended_Link2": new_link}}
    recommendations_df.update_one(query, new_values)

    query = {"Recommended_Book3": book_title}
    new_values = {"$set": {"Recommended_Link3": new_link}}
    recommendations_df.update_one(query, new_values)

    query = {"Recommended_Book4": book_title}
    new_values = {"$set": {"Recommended_Link4": new_link}}
    recommendations_df.update_one(query, new_values)

id = "65c8e6a21fe0b907adc2ab57"
new_link = "http://127.0.0.1:5001/open_book/" + id
filename = "Free"
update_link(filename, new_link)

