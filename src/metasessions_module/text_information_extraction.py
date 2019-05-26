import argparse
import logging
import os
import sys
import nltk


from nltk import pos_tag, word_tokenize
from nltk.stem import SnowballStemmer
from tqdm import tqdm
from pymystem3 import Mystem


sys.path.append(os.pardir)
sys.path.append(os.path.join(os.pardir, os.pardir))


from src.metasessions_module.config import *
from src.metasessions_module.text_utils import load_text


nltk.download('punkt')

logFormatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
rootLogger = logging.getLogger()

fileHandler = logging.FileHandler(os.path.join('logs', 'text_information_extraction.log'), 'a')
fileHandler.setFormatter(logFormatter)
rootLogger.addHandler(fileHandler)

consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(logFormatter)
rootLogger.addHandler(consoleHandler)
rootLogger.setLevel(logging.INFO)
log_step = 100000


def extract_verbs(book_ids):
    verbs = set()
    # stemmer = SnowballStemmer('russian')
    mystem = Mystem()
    for book_id in book_ids:
        logging.info('Extracting verbs from book {}'.format(book_id))
        text = load_text(book_id)
        # tokens = word_tokenize(text)
        # tagged_tokens = pos_tag(tokens, lang='rus')
        # for (word, tag), _ in zip(tagged_tokens, tqdm(range(len(tagged_tokens)))):
        #     if tag == 'V':
        #         verbs.add(stemmer.stem(word))
        tokens = mystem.analyze(text)
        for info in tqdm(tokens):
            if 'analysis' not in info or len(info['analysis']) == 0:
                continue
            info = info['analysis'][0]
            if 'gr' in info and info['gr'].split(',')[0] == 'V' and 'lex' in info:
                verbs.add(info['lex'])

    with open(os.path.join('resources', 'vocabulary', 'verbs', 'all.csv'), 'w') as file:
        for verb in sorted(verbs):
            file.write(verb + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--extract_verbs', action='store_true', help='Extract all the nouns from BOOKS texts')

    args = parser.parse_args()
    if args.extract_verbs:
        extract_verbs(list(BOOKS.values()))
