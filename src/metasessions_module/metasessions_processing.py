import pickle
import os

from .utils import connect_to_mongo_database


BOOKS = {'The Fault in Our Stars': 266700, 'Fifty Shades of Grey': 210901}


def load_sessions(book_id):
    with open(os.path.join('resources', '{book_id}.pkl'.format(book_id=book_id)), 'rb') as file:
        return pickle.load(file)


def save_sessions(book_ids):
    print('Books loading start')
    print('Books: ' + ', '.join([str(book_id) for book_id in book_ids]))
    db = connect_to_mongo_database('sessions')
    db_work = connect_to_mongo_database('bookmate_work')
    sessions = list(db['sessions'].find({"$or": [{'book_id': int(book_id)} for book_id in book_ids]}))
    print('Books loading finished')
    for book_id in book_ids:
        book_sessions = [session for session in sessions if session['book_id'] == book_id]
        db_work['sessions_{book_id}'.format(book_id=book_id)].insert_many(book_sessions)
    for book_id in book_ids:
        book_sessions = [session for session in sessions if session['book_id'] == book_id]
        with open(os.path.join('resources', '{book_id}.pkl'.format(book_id=book_id)), 'wb') as file:
            pickle.dump(book_sessions, file)
        print('book {book_id}: done'.format(book_id=book_id))


def save_book_session(book_id):
    save_sessions([book_id])


def get_book_documents_stats(book_id):
    collection_name = 'sessions_{book_id}'.format(book_id=book_id)
    db = connect_to_mongo_database('bookmate_work')
    document_ids = db[collection_name].distinct('document_id')
    results = {}
    book_sessions = load_sessions(book_id)
    for document_id in document_ids:
        document_sessions = [session for session in book_sessions if session['document_id'] == int(document_id)]
        users = list(set([session['user_id'] for session in document_sessions]))
        results[str(document_id)] = {'users_count': len(users),
                                     'users': users,
                                     'sessions_count': len(document_sessions),
                                     'sessions': document_sessions}
    # db['stats_{book_id}'.format(book_id=book_id)] = results  TODO:
    return results


def get_metasessions(book_id, cond):
    pass


def get_items(document_id):
    db = connect_to_mongo_database('bookmate')
    return list(db['items'].find({'document_id': document_id}))


if __name__ == '__main__':
    # save_sessions(BOOKS.values())
    # for title, book_id in BOOKS.items():
    #     res = get_book_documents_stats(book_id)
    #     print(title, end=os.linesep+os.linesep)
    #     total_res = []
    #     for doc_id, stats in res.items():
    #         total_res.append({
    #             'document': doc_id,
    #             'users number': stats['users_count'],
    #             'sessions number': stats['sessions_count']})
    #     total_res.sort(key=lambda record: record['sessions number'])
    #     print('document id,users number,sessions number')
    #     for result in total_res:
    #         print(','.join([str(result['document']), str(result['users number']), str(result['sessions number'])]))
    for title, book_id in BOOKS.items():
        res = get_book_documents_stats(book_id)
        for doc_id in res.keys():
            print(','.join([str(title), str(doc_id), str(len(get_items(int(doc_id))))]))
