import os
import sys
import argparse

from bert_serving.client import BertClient
from tqdm import tqdm

sys.path.append(os.pardir)
sys.path.append(os.path.join(os.pardir, os.pardir))
sys.path.append(os.path.join(os.pardir, os.pardir, os.pardir))

from src.metasessions_module.text_utils import load_book_fragments
from src.metasessions_module.utils import save_via_pickle, load_from_pickle


def encode_fragments(texts, with_output=True):
    bert_client = BertClient()
    if not with_output:
        return bert_client.encode(texts)
    result = []
    for text in tqdm(texts):
        result.append(bert_client.encode([text])[0])
    return result


def encode_book(book):
    texts = load_book_fragments(book)
    return encode_fragments(texts)


def get_embeddings_path(book, max_sequence_len):
    output_path = os.path.join('resources', 'bert_embeddings')
    os.makedirs(output_path, exist_ok=True)
    return os.path.join(output_path, f'{book}_{max_sequence_len}.pkl')


def save_embeddings(book, max_sequence_len=512):
    embeddings = encode_book(book)
    save_via_pickle(embeddings, get_embeddings_path(book, max_sequence_len))
    return embeddings


def get_embeddings(book, max_sequence_len=512):
    lookup_path = get_embeddings_path(book, max_sequence_len)
    if not os.path.exists(lookup_path):
        embeddings = save_embeddings(book, max_sequence_len)
    else:
        embeddings = load_from_pickle(lookup_path)
    return embeddings


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--book_id', type=int, required=True)
    args = parser.parse_args()
    book_id = args.book_id
    save_embeddings(book_id)
