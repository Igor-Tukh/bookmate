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


def verify_user_speed(sessions_collection, user_id):
    db = connect_to_mongo_database(BOOKS_DB)
    user_sessions = db[sessions_collection].find({'user_id': user_id,
                                                  'category': 'normal'})
    avr_abs_speed, avr_speed, sessions_num = 0, 0, user_sessions.count()
    for session in user_sessions:
        avr_abs_speed += session['abs_speed']
        avr_speed += session['speed']
    print ('User [%d]: avr_speed is [%.3f], avr_abs_speed is [%.3f]' % (user_id, avr_speed/sessions_num, avr_abs_speed/sessions_num))


books = ['210901']
for book_id in books:
    db = connect_to_mongo_database(BOOKS_DB)
    users = db['%s_target' % book_id].find().distinct('user_id')
    for user in users:
        verify_user_speed('%s_target' % book_id, user)
    # get_users_info('%s_target' % book_id, book_id)