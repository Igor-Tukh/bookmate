import logging
import os


def get_text_from_txt(filepath):
    logging.info('Loading file {}'.format(filepath))
    with open(filepath, 'r') as file:
        return file.read()


def load_text(book_id, document_id):
    logging.info('Loading text for document {} of book {}'.format(document_id, book_id))
    books_path = os.path.join('resources', 'books')
    document_path = os.path.join(books_path, '{}_{}.txt'.format(book_id, document_id))
    book_path = os.path.join(books_path, '{}.txt'.format(book_id))
    if os.path.isfile(document_path):
        logging.info('{} found'.format(document_path))
        return get_text_from_txt(document_path)
    elif os.path.isfile(book_path):
        logging.info('{} found'.format(book_path))
        return get_text_from_txt(book_path)
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
