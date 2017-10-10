from pymongo import MongoClient

BOOKS_DB = 'bookmate'


def connect_to_mongo_database(db):
    client = MongoClient('localhost', 27017)
    db = client[db]
    return db


def split_users(book_id):
    db = connect_to_mongo_database(BOOKS_DB)
    book_users = db[book_id].find().distinct('user_id')

    for user in book_users:
        user_profile = db['users'].find_one({'id': user})
        if user_profile is None:
            continue
        user_sessions = db[book_id].find({'user_id': user})

        for session in user_sessions:
            if user_profile['gender'] == 'm':
                db['%s_male' % book_id].insert(session)
            elif user_profile['gender'] == 'f':
                db['%s_female' % book_id].insert(session)

            if user_profile['sub_level'] == 0:
                db['%s_free' % book_id].insert(session)
            else:
                db['%s_paid' % book_id].insert(session)


books = ['2289', '2207', '2206', '2543', '135089']
for book_id in books:
    split_users(book_id)