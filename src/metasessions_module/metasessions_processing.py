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
    calculate_session_percents, load_user_sessions, save_user_sessions_speed
from metasessions_module.user_utils import save_users, save_books_users_sessions, save_common_users, get_common_users, \
    get_user_document_id, load_users, get_users_books_amount
from metasessions_module.text_utils import load_chapters, load_text, get_chapter_percents
from metasessions_module.item_utils import get_items
from tqdm import tqdm
from enum import Enum


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


class ReadingStyle(Enum):
    SCANNING = 1
    SKIMMING = 2
    NORMAL = 3
    DETAILED = 4

    def get_color(self):
        if self == ReadingStyle.SCANNING:
            return 'ro'
        elif self == ReadingStyle.SKIMMING:
            return 'go'
        elif self == ReadingStyle.NORMAL:
            return 'bo'
        elif self == ReadingStyle.DETAILED:
            return 'yo'

    @staticmethod
    def get_reading_style_by_speed(speed,
                                   scanning_lower_thrshold=3500.0,
                                   skimming_lower_threshold=2500,
                                   normal_lower_threshold=1150):
        if speed > scanning_lower_thrshold:
            return ReadingStyle.SCANNING
        elif speed > skimming_lower_threshold:
            return ReadingStyle.SKIMMING
        elif speed > normal_lower_threshold:
            return ReadingStyle.NORMAL
        return ReadingStyle.DETAILED


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


def get_metasessions_breaks(book_id, document_id, user_id, break_time_seconds=1800):
    logging.info('Looking for metasessions break for user {} and document {} of book {}'.format(user_id,
                                                                                                document_id,
                                                                                                book_id))
    user_sessions = load_user_sessions(book_id, document_id, user_id)
    if len(user_sessions) == 0:
        return []
    user_sessions.sort(key=lambda session: date_from_timestamp(session['read_at']))
    previous_session = user_sessions[0]
    breaks = []
    for session in user_sessions[1:]:
        time = (date_from_timestamp(session['read_at']) - date_from_timestamp(previous_session['read_at'])) \
            .total_seconds()
        if time > break_time_seconds:
            breaks.append(session)
        previous_session = session
    return breaks


def save_metasessions_by_reading_style(book_id, document_id, user_id):
    logging.info('Start saving metasessions by reading style for user {} and document {} of book {}'.format(user_id,
                                                                                                            document_id,
                                                                                                            book_id))
    user_sessions = load_user_sessions(book_id, document_id, user_id)
    if len(user_sessions) == 0:
        return
    if 'speed' not in user_sessions[0]:
        save_user_sessions_speed(book_id, document_id, user_id)
    user_sessions.sort(key=lambda session: date_from_timestamp(session['read_at']))
    metasessions = [[user_sessions[0]]]
    previous_reading_style = ReadingStyle.get_reading_style_by_speed(user_sessions[0]['speed'])
    for session in user_sessions[1:]:
        current_reading_style = ReadingStyle.get_reading_style_by_speed(session['speed'])
        if current_reading_style == previous_reading_style:
            metasessions[-1].append(session)
        else:
            previous_reading_style = current_reading_style
            metasessions.append([session])

    logging.info('Total found {} metasessions'.format(len(metasessions)))
    metasessions_path = os.path.join('resources', 'metasessions_style', '{}_{}_{}.pkl'.format(book_id,
                                                                                              document_id,
                                                                                              user_id))
    with open(metasessions_path, 'wb') as file:
        pickle.dump(metasessions, file)
    logging.info('Metasessions by style of document {} for user {} saved to {}'.format(document_id,
                                                                                       user_id,
                                                                                       metasessions_path))


def save_metasessions_by_deviant_percent(book_id, document_id, user_id, deviant_percent=50, first_sessions_amount=3):
    logging.info('Start saving metasessions by reading style for user {} and document {} of book {}'.format(user_id,
                                                                                                            document_id,
                                                                                                            book_id))
    user_sessions = load_user_sessions(book_id, document_id, user_id)
    if len(user_sessions) == 0:
        return
    if 'speed' not in user_sessions[0]:
        save_user_sessions_speed(book_id, document_id, user_id)
    user_sessions.sort(key=lambda session: date_from_timestamp(session['read_at']))
    metasessions = [[user_sessions[0]]]
    for session in user_sessions[1:]:
        if len(metasessions[-1]) < 3:
            metasessions[-1].append(session)
        else:
            base_speed = sum([metasessions[-1][i]['speed']
                              for i in range(first_sessions_amount)]) / first_sessions_amount
            big_skip = abs(session['book_from'] - metasessions[-1][-1]['book_from']) > 1.0
            if not big_skip and session['speed'] > 0.00001 \
                    and abs(base_speed - session['speed']) / session['speed'] < deviant_percent / 100:
                metasessions[-1].append(session)
            else:
                metasessions.append([session])

    logging.info('Total found {} metasessions'.format(len(metasessions)))
    metasessions_path = os.path.join('resources', 'metasessions_deviant_percent', '{}_{}_{}.pkl'.format(book_id,
                                                                                                        document_id,
                                                                                                        user_id))
    with open(metasessions_path, 'wb') as file:
        pickle.dump(metasessions, file)
    logging.info('Metasessions by style of document {} for user {} saved to {}'.format(document_id,
                                                                                       user_id,
                                                                                       metasessions_path))


def get_metasessions_by_reading_style(book_id, document_id, user_id):
    metasessions_path = os.path.join('resources', 'metasessions_style', '{}_{}_{}.pkl'.format(book_id,
                                                                                              document_id,
                                                                                              user_id))
    if not os.path.isfile(metasessions_path):
        logging.error('Unable to find metasessions by reading style of document {} for user {}'.format(document_id,
                                                                                                       user_id))
        save_metasessions_by_reading_style(book_id, document_id, user_id)
        return

    with open(metasessions_path, 'rb') as file:
        return pickle.load(file)


def get_metasessions_by_deviant_percent(book_id, document_id, user_id):
    metasessions_path = os.path.join('resources', 'metasessions_deviant_percent', '{}_{}_{}.pkl'.format(book_id,
                                                                                                        document_id,
                                                                                                        user_id))
    if not os.path.isfile(metasessions_path):
        logging.error('Unable to find metasessions by reading style of document {} for user {}'.format(document_id,
                                                                                                       user_id))
        save_metasessions_by_deviant_percent(book_id, document_id, user_id)
        return

    with open(metasessions_path, 'rb') as file:
        return pickle.load(file)


def visualize_metasessions_by_reading_style(book_id, document_id, user_id):
    plt.clf()
    plt.xlabel('Book percent')
    plt.ylabel('Session speed')
    plt.title('Metasessions visualization')
    plt.ylim(50.0, 6000.0)
    plt.xlim(0.0, 100.0)
    metasessions = get_metasessions_by_reading_style(book_id, document_id, user_id)
    chapters_lens = get_chapter_percents(book_id, document_id)
    for chapter_len in chapters_lens[:-1]:
        plt.axvline(x=chapter_len, color='black', linestyle='--', linewidth=0.5)
    for metasession in metasessions:
        avg_speed = sum([session['speed'] for session in metasession]) / len(metasession)
        x = [session['book_from'] for session in metasession]
        y = [avg_speed for _ in metasession]
        plt.plot(x, y, ReadingStyle.get_reading_style_by_speed(metasession[0]['speed']).get_color(), markersize=2)
    plt.savefig(os.path.join('resources', 'plots', 'reading_style', '{}_{}_{}_metasessions_by_reading_style.png')
                .format(book_id, document_id, user_id))


def visualize_metasessions_by_deviant_percent(book_id, document_id, user_id):
    metasessions = get_metasessions_by_deviant_percent(book_id, document_id, user_id)
    chapters_lens = get_chapter_percents(book_id, document_id)
    for chapter_len in chapters_lens[:-1]:
        plt.axvline(x=chapter_len, color='black', linestyle='--', linewidth=0.5)
    for metasession in metasessions:
        avg_speed = sum([session['speed'] for session in metasession]) / len(metasession)
        x = [session['book_from'] for session in metasession]
        y = [avg_speed for _ in metasession]
        plt.plot(x, y)
    plt.savefig(os.path.join('resources', 'plots', 'deviant_percent', '{}_{}_{}_metasessions_by_deviant_percent.png')
                .format(book_id, document_id, user_id))


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
    logging.info('Metasessions of document {} for user {} saved to {}'.format(document_id, user_id, metasessions_path))


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
    plt.ylim(50.0, 5000.0)
    plt.xlim(0.0, 100.0)
    metassesions = get_metassesions(book_id, document_id, user_id)
    for metassesion in metassesions:
        x = [session['book_from'] for session, _ in metassesion]
        y = [speed for _, speed in metassesion]
        plt.plot(x, y)
    plt.savefig(os.path.join('resources', 'plots', '{}_{}_{}_metasessions.png').format(book_id, document_id, user_id))


def upload_good_users(book_id):
    users_path = os.path.join('resources', 'users', '{}_good_users_id_amount.pkl'.format(book_id))
    if os.path.isfile(users_path):
        with open(users_path, 'rb') as file:
            return pickle.load(file)
    else:
        logging.error('Unable to upload good users for book {}'.format(book_id))


def get_user_selection(book_id):
    output_path = os.path.join('resources', 'users', '{}_users_selection.pkl'.format(book_id))
    if os.path.isfile(output_path):
        with open(output_path, 'rb') as file:
            return pickle.load(file)
    logging.error('Unable to load users selection for book {}'.format(book_id))
    return []


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
    parser.add_argument('--visualize_metasessions_by_style', help='Visualize metasessions by reading style',
                        action='store_true')
    parser.add_argument('--visualize_metasessions_by_deviant_percent', help='Visualize metasessions by reading style',
                        action='store_true')
    parser.add_argument('--prepare_users', help='Find users which red a lot of books',
                        action='store_true')
    parser.add_argument('--select_users', help='Select users with the most amount of sessions',
                        action='store_true')
    parser.add_argument('--save_speed', help='Save speed for users sessions',
                        action='store_true')

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
    if args.visualize_metasessions_by_style:
        for book_id in BOOKS.values():
            for user_id in tqdm(get_user_selection(book_id)):
                logging.info('Visualizing metasessions by speed of user {}'.format(user_id))
                document_id = get_user_document_id(book_id, user_id)
                if document_id not in DOCUMENTS[book_id]:
                    continue
                visualize_metasessions_by_reading_style(book_id, document_id, user_id)
    if args.visualize_metasessions_by_deviant_percent:
        for book_id in BOOKS.values():
            for user_id in tqdm(get_user_selection(book_id)):
                logging.info('Visualizing metasessions by deviant percent of user {}'.format(user_id))
                document_id = get_user_document_id(book_id, user_id)
                if document_id not in DOCUMENTS[book_id]:
                    continue
                visualize_metasessions_by_deviant_percent(book_id, document_id, user_id)
    if args.save_speed:
        for book_id in BOOKS.values():
            for user_id in tqdm(get_user_selection(book_id)):
                logging.info('Saving speed for sessions of user {}'.format(user_id))
                document_id = get_user_document_id(book_id, user_id)
                save_user_sessions_speed(book_id, document_id, user_id)
    if args.save_metasessions:
        for book_id in BOOKS.values():
            for user_id in tqdm(get_user_selection(book_id)):
                logging.info('Saving metasessions for user {}'.format(user_id))
                document_id = get_user_document_id(book_id, user_id)
                save_metasessions_by_reading_style(book_id, document_id, user_id)
                save_metasessions_by_deviant_percent(book_id, document_id, user_id)
    if args.prepare_users:
        good_users = {}
        for book_id in BOOKS.values():
            book_users = load_users(book_id)
            good_users[book_id] = []
            iter = 0
            for user_id in tqdm(book_users):
                if iter % 100 == 0:
                    output_path = os.path.join('resources', 'users', '{}_good_users_id_amount.pkl'.format(book_id))
                    with open(output_path, 'wb') as file:
                        pickle.dump(good_users[book_id], file)
                sessions_amount = 0
                for document_id in DOCUMENTS[book_id]:
                    sessions_amount += len(load_user_sessions(book_id, document_id, user_id))
                if sessions_amount < 100:
                    continue
                if user_id == '':
                    continue
                books_amount = get_users_books_amount(user_id)
                if books_amount >= 10:
                    good_users[book_id].append((user_id, books_amount))
                    logging.info('Found good user {}, who red {} books'.format(user_id, books_amount))
                iter += 1
    if args.select_users:
        users_for_book = {}
        for book_id in BOOKS.values():
            users = upload_good_users(book_id)
            users_for_book[book_id] = []
            for user_id, amount in users:
                document_id = get_user_document_id(book_id, user_id)
                sessions_amount = len(load_user_sessions(book_id, document_id, user_id))
                users_for_book[book_id].append((user_id, sessions_amount))
            users_for_book[book_id].sort(key=lambda x: -x[1])
            output_path = os.path.join('resources', 'users', '{}_users_selection.pkl'.format(book_id))
            logging.info('Total found {} good users for book {}, we will select 100'
                         .format(len(users_for_book[book_id]), book_id))
            with open(output_path, 'wb') as file:
                pickle.dump([x[0] for x in users_for_book[book_id][:100]], file)
