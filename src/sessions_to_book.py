from pymongo import MongoClient

def connect_to_mongo_database(book_id) -> None:
    client = MongoClient('localhost', 27017)
    db = client.bookmate
    return db


def process_sessions_to_pages(book_id, user_id):
    db = connect_to_mongo_database(book_id)
    sessions = db[user_id]
    pages = db[book_id + '_window']

    for session in sessions:
        page_session = pages.find({ 'to_percent': {'$lte': session['to_percent']},
                                    'from_percent': {'$gte': session['from_percent']} })