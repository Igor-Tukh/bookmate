import logging
import os
import csv


from collections import defaultdict
from tqdm import tqdm
from src.metasessions_module.user_utils import get_user_document_id, load_users
from src.metasessions_module.sessions_utils import load_user_sessions, load_document_sessions


def save_book_documents_stats(book_id):
    documents_stats_file = os.path.join('resources', 'documents', '{}.csv'.format(book_id))
    if os.path.isfile(documents_stats_file):
        logging.info('{} found, book {} skipped'.format(documents_stats_file, book_id))
        return
    logging.info('Saving documents stats for book {}'.format(book_id))
    users = []
    try:
        users = load_users(book_id)
    except ValueError:
        logging.error('Can\'t find users for book {}'.format(book_id))
    documents = []
    for user_id in tqdm(users):
        documents.append(get_user_document_id(book_id, user_id))
    users_count = defaultdict(lambda: 0)
    sessions_count = {}
    logging.info('Start selection stats for book {} documents'.format(book_id))
    for document in tqdm(documents):
        users_count[document] += 1
    users_count = dict(users_count)
    for document in tqdm(users_count.keys()):
        sessions_count[document] = len(load_document_sessions(book_id, document))

    if len(users_count) == 0:
        logging.info('No documents found for book {}'.format(book_id))
        return

    with open(documents_stats_file, 'w') as stats_file:
        results = [{'book id': book_id,
                    'document id': key,
                    'users amount': value,
                    'sessions amount': sessions_count[key]} for key, value in users_count.items()]
        writer = csv.DictWriter(stats_file, results[0].keys())
        writer.writeheader()
        writer.writerows(results)
        logging.info('Book {} documents stats saved to {}'.format(book_id, documents_stats_file))
