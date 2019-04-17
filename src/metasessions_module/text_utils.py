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
    return load_text(book_id, document_id).split('---')
