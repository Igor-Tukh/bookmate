import logging
import os
import sys
import numpy as np

from src.metasessions_module.config import get_book_fragments_path
from src.metasessions_module.sessions_utils import load_user_sessions
from src.metasessions_module.user_utils import get_good_users_info, get_user_document_id
from tqdm import tqdm
from collections import defaultdict

sys.path.append(os.pardir)
sys.path.append(os.path.join(os.pardir, os.pardir))


def get_text_from_txt(filepath):
    logging.info('Loading file {}'.format(filepath))
    with open(filepath, 'r') as file:
        return file.read()


def load_text(book_id, document_id=None):
    logging.info('Loading text for document {} of book {}'.format(document_id, book_id))
    books_path = os.path.join('resources', 'books')
    document_path = os.path.join(books_path, '{}_{}.txt'.format(book_id, document_id)) \
        if document_id is not None else ''
    book_path = os.path.join(books_path, '{}.txt'.format(book_id))
    if os.path.isfile(book_path):
        logging.info('{} found'.format(book_path))
        return get_text_from_txt(book_path)
    elif os.path.isfile(document_path):
        logging.info('{} found'.format(document_path))
        return get_text_from_txt(document_path)
    else:
        logging.info('Nothing found for document {} of book {}'.format(document_id, book_id))

    return None


def load_chapters(book_id, document_id):
    chapters = load_text(book_id, document_id).split('-----')
    chapters[1] = chapters[0] + chapters[1]
    logging.info('Found {} chapters for document_id {} of book {}'.format(len(chapters) - 1, document_id, book_id))
    return chapters[1:]


def get_chapter_percents(book_id, document_id):
    chapters = load_chapters(book_id, document_id)
    chapters_lens = [len(chapter) for chapter in chapters]
    total_len = sum(chapters_lens)
    chapter_percents = []
    current_len = 0
    for chapter_len in chapters_lens:
        current_len += chapter_len
        chapter_percents.append((100.0 * current_len) / total_len)
    return chapter_percents


def is_text_character(letter):
    return letter.isalpha() or letter.isalnum()


def get_the_most_popular_screen_size(book_id):
    user_ids = list(get_good_users_info(book_id).keys())
    size_frequencies = defaultdict(lambda: 0)
    for user_id, _ in zip(user_ids, tqdm(range(len(user_ids)))):
        document_id = get_user_document_id(book_id, user_id)
        sessions = load_user_sessions(book_id, document_id, user_id)
        for session in sessions:
            size_frequencies[int(session['size'])] += 1
    size_stats = list(size_frequencies.items())
    size_stats.sort(key=lambda value: -value[1])
    return int(np.mean(np.array(size_stats)[:100, 0]))


def save_the_most_popular_screen_size_split(book_id, batches_amount=400):
    text = load_text(book_id)
    # slice_size = get_the_most_popular_screen_size(book_id)
    slice_size = int(len(text) / 400)
    print('Slice size: {}'.format(slice_size))
    start_index = 0
    while start_index < len(text):
        finish_index = start_index + slice_size
        while finish_index < len(text) and text[finish_index] != ' ':
            finish_index += 1
        with open(os.path.join('resources', 'books', 'splitted_text', '{}_{}_{}.txt'
                .format(book_id, start_index, finish_index)), 'w') as text_file:
            text_file.write(text[start_index:finish_index])
        start_index = finish_index


def get_split_text_borders(book_id):
    text_path = os.path.join('resources', 'books', str(book_id))
    if not os.path.exists(text_path):
        logging.info(f'Split text for the book {book_id} not found')
        return None
    chunks = []
    for index in range(len(os.listdir(text_path))):
        with open(os.path.join(text_path, f'text_{index}.txt'), 'r') as text_file:
            chunks.append(text_file.read())
    total_len = sum([len(chunk) for chunk in chunks])
    borders = []
    current_len = 0
    for chunk in chunks:
        current_len += len(chunk)
        borders.append(1. * current_len / total_len)
    return borders


def load_book_fragments(book_id):
    dir_path = get_book_fragments_path(book_id)
    filenames = os.listdir(dir_path)
    fragments = ['' for _ in range(len(filenames))]
    for filename in filenames:
        text_ind = filename.split('.')[0].split('_')[1]
        with open(os.path.join(dir_path, filename), 'r') as text_file:
            fragments[int(text_ind)] = text_file.read()
    return fragments


def load_chapters_mask(book_id, chapter_label='Глава'):
    texts = load_book_fragments(book_id)
    chapters_mask = np.zeros(len(texts), dtype=np.bool)
    for text_ind, text in enumerate(texts):
        if chapter_label in text:
            chapters_mask[text_ind] = True
            if text_ind >= 1:
                chapters_mask[text_ind - 1] = True
    return chapters_mask


if __name__ == '__main__':
    # save_the_most_popular_screen_size_split(135089)
    print(load_chapters_mask(210901))
    print(np.sum(load_chapters_mask(210901), dtype=np.int))
