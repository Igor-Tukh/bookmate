from pymongo import MongoClient

BOOKS_DB = 'bookmate'


def connect_to_mongo_database(db):
    client = MongoClient('localhost', 27017)
    db = client[db]
    return db


def get_users_info(sessions_collection, book_id):
    db = connect_to_mongo_database(BOOKS_DB)
    users = db[sessions_collection].find().distinct('user_id')

    for user_id in users:
        user_sessions = db[sessions_collection].find({'user_id': user_id})
        db['%s_users' % book_id].update({'_id': user_id},
                                        {'$set': {'sessions': user_sessions.count()}})


books = ['2206', '2207', '2289', '2543', '135089']
for book_id in books:
    get_users_info('%s_target' % book_id, book_id)