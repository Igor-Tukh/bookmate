import logging
import argparse
import os
import sys
import pickle
import matplotlib.pyplot as plt


sys.path.append(os.pardir)
sys.path.append(os.path.join(os.pardir, os.pardir))

from metasessions_module.utils import connect_to_mongo_database, date_from_timestamp
from metasessions_module.sessions_utils import load_sessions, save_sessions, save_book_sessions, \
    calculate_session_percents, load_user_sessions
from metasessions_module.user_utils import save_users, save_books_users_sessions, save_common_users, get_common_users, \
    get_user_document_id, load_users
from metasessions_module.text_utils import load_chapters, load_text
from metasessions_module.item_utils import get_items
from tqdm import tqdm


logFormatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
rootLogger = logging.getLogger()

fileHandler = logging.FileHandler(os.path.join('logs', 'metasessions_processing.log'), 'a')
fileHandler.setFormatter(logFormatter)
rootLogger.addHandler(fileHandler)

consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(logFormatter)
rootLogger.addHandler(consoleHandler)
rootLogger.setLevel(logging.INFO)
log_step = 100000

BOOKS = {'The Fault in Our Stars': 266700, 'Fifty Shades of Grey': 210901}
DOCUMENTS = {210901: [1143157, 1416430, 1311858], 266700: [969292, 776328, 823395]}


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


def save_metasessions(book_id, document_id, user_id):
    user_sessions = load_user_sessions(book_id, document_id, user_id)
    logging.info('Found {} sessions for document_id {} and user_id {}'.format(len(user_sessions), document_id, user_id))
    user_sessions.sort(key=lambda session: date_from_timestamp(session['read_at']))
    if len(user_sessions) == 0:
        return
    metasessions = [[(user_sessions[0], 0)]]
    for ind, session in enumerate(user_sessions[1:], 1):
        previous_session = user_sessions[ind - 1]
        time = (date_from_timestamp(session['read_at']) - date_from_timestamp(previous_session['read_at'])) \
            .total_seconds()
        if time < 0.01:
            speed = 3000  # TODO inf
        else:
            speed = previous_session['size'] * 60 / time
        if speed < 100:  # TODO threshold
            metasessions.append([(session, speed)])
        else:
            metasessions[-1].append((session, speed))
    logging.info('Total found {} metasessions'.format(len(metasessions)))
    metasessions_path = os.path.join('resources', 'metasessions', '{}_{}_{}.pkl'.format(book_id, document_id, user_id))
    with open(metasessions_path, 'wb') as file:
        pickle.dump(metasessions, file)
    logging.info('Metassesions of document {} for user {} saved to {}'.format(document_id, user_id, metasessions_path))


def get_metassesions(book_id, document_id, user_id):
    metasessions_path = os.path.join('resources', 'metasessions', '{}_{}_{}.pkl'.format(book_id, document_id, user_id))
    if not os.path.isfile(metasessions_path):
        logging.error('Unable to find metasessions of document {} for user {}'.format(document_id, user_id))
    with open(metasessions_path, 'rb') as file:
        return pickle.load(file)


def visualize_metassesions(book_id, document_id, user_id):
    plt.clf()
    plt.xlabel('Book percent')
    plt.ylabel('Session speed')
    plt.title('Metasessions visualization')
    plt.ylim(50.0, 4000.0)
    plt.xlim(0.0, 100.0)
    metassesions = get_metassesions(book_id, document_id, user_id)
    for metassesion in metassesions:
        x = [session['book_from'] for session, _ in metassesion]
        y = [speed for _, speed in metassesion]
        plt.plot(x, y)
    plt.savefig(os.path.join('resources', 'plots', '{}_{}_{}_metasessions.png').format(book_id, document_id, user_id))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_sessions', help='Save BOOKS sessions', action='store_true')
    parser.add_argument('--save_users', help='Save BOOKS users', action='store_true')
    parser.add_argument('--save_users_sessions', help='Save sessions for users of BOOKS', action='store_true')
    parser.add_argument('--calculate_session_percents', help='Calculate book_from and book_to', action='store_true')
    parser.add_argument('--save_common_users', help='Save common users of BOOKS', action='store_true')
    parser.add_argument('--process_items', help='process DOCUMENTS items', action='store_true')
    parser.add_argument('--save_metasessions', help='Save BOOKS metasessions', action='store_true')
    parser.add_argument('--find_the_best_users', help='Find the best users', action='store_true')


    args = parser.parse_args()
    if args.save_sessions:
        save_sessions(BOOKS.values())
    if args.save_users:
        save_users(BOOKS.values())
    if args.save_users_sessions:
        save_books_users_sessions(BOOKS.values())
    if args.save_common_users:
        save_common_users(list(BOOKS.values()))
    if args.calculate_session_percents:
        for book_id in BOOKS.values():
            calculate_session_percents(book_id, DOCUMENTS[book_id])
    if args.find_the_best_users:
        for book_id in BOOKS.values():
            users = load_users(book_id)[100:]
            for user_id, _ in zip(users, tqdm(range(len(users)))):
                user_document_id = get_user_document_id(book_id, user_id)
                if user_document_id not in DOCUMENTS[book_id]:
                    continue
                if user_id is '':
                    continue
                user_sessions = load_user_sessions(book_id, user_document_id, user_id)
                if len(user_sessions) < 500:
                    continue
                print(book_id, user_document_id, user_id, len(user_sessions))
    if args.process_items:
        for book_id in BOOKS.values():
            for document_id in DOCUMENTS[book_id]:
                print('Book {}, document {}'.format(book_id, document_id))
                chapters = load_chapters(book_id, document_id)
                text = load_text(book_id, document_id)
                items = get_items(document_id)
                print('Items:')
                for item in items:
                    print('{},{},{},{}'.format(book_id, document_id, item['position'], item['_to'] - item['_from']))
                print('Chapters')
                for ind, chapter in enumerate(chapters):
                    print('{},{}'.format(ind + 1, 100.0 * len(chapter) / len(text)))

    if args.save_metasessions:
        # users = get_common_users(list(BOOKS.values()))[:10]
        # logging.info('Found common users: {}'.format(str(users)))
        # for book_id in BOOKS.values():
        #     for user_id in users:
        #         if user_id is '':
        #             continue
        #         user_document_id = get_user_document_id(book_id, user_id)
        #         save_metasessions(book_id, user_document_id, user_id)
        #         visualize_metassesions(book_id, user_document_id, user_id)
        save_metasessions(266700, 969292, 393331)
        visualize_metassesions(266700, 969292, 393331)
        save_metasessions(210901, 1143157, 1966674)
        visualize_metassesions(210901, 1143157, 1966674)
        save_metasessions(210901, 1311858, 2228827)
        visualize_metassesions(210901, 1311858, 2228827)
        save_metasessions(210901, 1143157, 1966770)
        visualize_metassesions(210901, 1143157, 1966770)
        save_metasessions(210901, 1143157, 1966782)
        visualize_metassesions(210901, 1143157, 1966782)

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
    # for title, book_id in BOOKS.items():
    #     res = get_book_documents_stats(book_id)
    #     for doc_id in res.keys():
    #         print(','.join([str(title), str(doc_id), str(len(get_items(int(doc_id))))]))
