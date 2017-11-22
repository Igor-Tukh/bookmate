# -*- coding: utf-8 -*-
import glob
import os
import xml.etree.ElementTree as et
import re
import codecs
import json
import csv
import datetime
import subprocess
import multiprocessing

mystem_in_directory = 'mystem_in/'
mystem_out_directory = 'mystem_out/'

class AggregatedTextStats:
    def __init__(self, flibusta_id):
        self.flibusta_id = flibusta_id
        self.image_count = 0
        self.symbol_count = 0
        self.word_count = 0
        self.letter_in_word_count = 0
        self.sentence_count = 0
        self.paragraph_count = 0
        self.direct_speech_sentence_count = 0 #??
        self.comma_count = 0
        self.dash_count = 0
        self.verb_personal_count = 0
        self.verb_infinitive_count = 0
        self.gerund_count = 0
        self.participle_count = 0
        self.noun_count = 0
        self.adjective_count = 0
        self.adverb_count = 0
        self.word_on_cap_letter_count = 0
        self.latin_word_count = 0
        self.unique_words = set()
        self.interrogative_sign_count = 0
        self.exclamatory_sign_count = 0
        self.three_dot_sign_count = 0

    @staticmethod
    def get_naive_features_fieldnames():
        return [
                'flibusta_id',
                'images',
                'symbols',
                'paragraphs'
            ]

    @staticmethod
    def get_mystem_features_fieldnames():
        return [
                'flibusta_id',
                'words',
                'ltrs_in_word',
                'sentences', 
                'dir_sp_sentences',
                'commas',
                'dashes',
                'pers_verbs',
                'inf_verbs',
                'gerungs',
                'participles',
                'nouns',
                'adjectives',
                'adverbs',
                'words_on_cap_ltr',
                'latin_words',
                'uniq_words',
                'interrogative',
                'exclamatory',
                'three_dots',
            ]

    def get_naive_features_row(self):
        return {
                'flibusta_id':self.flibusta_id,
                'images':self.image_count,
                'symbols':self.symbol_count,
                'paragraphs':self.paragraph_count
            }

    def get_mystem_features_row(self):
        return {
                'flibusta_id':self.flibusta_id,
                'words':self.word_count,
                'ltrs_in_word':self.letter_in_word_count,
                'sentences':self.sentence_count, 
                'dir_sp_sentences':self.direct_speech_sentence_count,
                'commas':self.comma_count,
                'dashes':self.dash_count,
                'pers_verbs':self.verb_personal_count,
                'inf_verbs':self.verb_infinitive_count,
                'gerungs':self.gerund_count,
                'participles':self.participle_count,
                'nouns':self.noun_count,
                'adjectives':self.adjective_count,
                'adverbs':self.adverb_count,
                'words_on_cap_ltr':self.word_on_cap_letter_count,
                'latin_words':self.latin_word_count,
                'uniq_words':len(self.unique_words),
                'interrogative':self.interrogative_sign_count,
                'exclamatory':self.exclamatory_sign_count,
                'three_dots':self.three_dot_sign_count
            }

def add_log(text):
    with open('fb2_processing_logs.txt', 'a') as f:
        f.write(text + '\n')

def get_el_text_and_image_count(element):
    text = element.text
    image_count = 0
    if text is None:
        text = ""
    for child in element:
        if child.tag in ['strong', 'strikethrough', 'poem', 'emphasis', 'cite'] and child.text is not None:
            text += child.text
        else:
            text += ' '
        if child.tag == 'image':
            image_count += 1
    return (text, image_count)

def process_paragraph_element(element, text_stats, mystem_in):
    text, image_count = get_el_text_and_image_count(element)
    mystem_in.write(text)
    mystem_in.write('\n')
    text_stats.image_count += image_count
    text_stats.symbol_count += len(text)
    text_stats.paragraph_count += 1

def contains_end_of_sentence(text):
    return '.' in text or '?' in text or '\n' in text or '!' in text or u'…' in text

def contains_comma(text):
    return ',' in text or ';' in text

def contains_dash(text):
    return '-' in text or u'—' in text

def get_end_of_sentence_type(text):
    if '?' in text:
        return 'interrogative'
    if '!' in text:
        return 'exclamatory'
    if '...' in text or u'…' in text:
        return 'three_dots'
    return 'usual'

latin_letters = set('qwertyuiopasdfghjklzxcvbnm')
def is_latin_word(text):
    for letter in text.lower():
        if letter not in latin_letters:
            return False
    return True

def process_mystemed_paragraph(text, text_stats):
    json_text = json.loads(text)
    last_token_was_end_of_sentence = False
    first_token = True
    for token in json_text:
        if token['text'] == '\\s':
            continue
        if 'analysis' in token:
            text_stats.word_count += 1
            text_stats.letter_in_word_count += len(token['text'])
            if len(token['analysis']) != 0:
                if 'gr' in token['analysis'][0]:
                    gramem = set().union(*(set(x.split('=')) for x in token['analysis'][0]['gr'].split(',')))
                    if 'V' in gramem:
                        if 'indic' in gramem or 'imper' in gramem:
                            text_stats.verb_personal_count += 1
                        if 'inf' in gramem:
                            text_stats.verb_infinitive_count += 1
                        if 'ger' in gramem:
                            text_stats.gerund_count += 1
                        if 'partcp' in gramem:
                            text_stats.participle_count += 1
                    if 'S' in gramem:
                        text_stats.noun_count += 1
                    if 'A' in gramem:
                        text_stats.adjective_count += 1
                    if 'ADV' in gramem:
                        text_stats.adverb_count += 1
                if 'lex' in token['analysis'][0]:
                    text_stats.unique_words.add(token['analysis'][0]['lex'])
            if (not last_token_was_end_of_sentence) and token['text'][0].isupper(): # check!
                text_stats.word_on_cap_letter_count += 1
            if is_latin_word(token['text']):
                text_stats.latin_word_count += 1 
            last_token_was_end_of_sentence = False
        else:
            if contains_end_of_sentence(token['text']):
                if not last_token_was_end_of_sentence:
                    text_stats.sentence_count += 1
                    end_of_sentence_type = get_end_of_sentence_type(token['text'])
                    if end_of_sentence_type == 'interrogative':
                        text_stats.interrogative_sign_count += 1
                    elif end_of_sentence_type == 'exclamatory':
                        text_stats.exclamatory_sign_count += 1
                    elif end_of_sentence_type == 'three_dots':
                        text_stats.three_dot_sign_count += 1
                last_token_was_end_of_sentence = True
            else:
                last_token_was_end_of_sentence = False
                if contains_comma(token['text']):
                    text_stats.comma_count += 1
                if contains_dash(token['text']):
                    if first_token:
                        text_stats.direct_speech_sentence_count += 1
                    text_stats.dash_count += 1
        first_token = False
 
def get_flibusta_id(filename):
    return int(filename[filename.index('/') + 1:filename.index('.')])

def build_mystem_in_file(flibusta_id):
    naive_stats = AggregatedTextStats(flibusta_id)
    try:
        element_tree = et.parse('fb2/{0}.fb2'.format(flibusta_id))
    except:
        print(flibusta_id)
        return naive_stats
    root = element_tree.getroot()
    xmlns = root.tag[:-11]
    with codecs.open('{0}{1}.mystem.in'.format(mystem_in_directory, flibusta_id), 'w', encoding='utf-8') as mystem_in:
        for p_element in root.iter(tag=xmlns + 'p'):
            process_paragraph_element(p_element, naive_stats, mystem_in)
    return naive_stats

def get_mystem_in_files():
    start = datetime.datetime.now()
    filenames = glob.glob('fb2/*.fb2')
    with open('naive_content_based_features.csv', 'w') as f:
        writer = csv.DictWriter(f, fieldnames = AggregatedTextStats.get_naive_features_fieldnames())
        writer.writeheader()
        for filename in filenames:
            flibusta_id = get_flibusta_id(filename)
            naive_stats = build_mystem_in_file(flibusta_id)
            writer.writerow(naive_stats.get_naive_features_row())
            add_log('{0}{1}.mystem.in is written'.format(mystem_in_directory, flibusta_id))
    finish = datetime.datetime.now()
    print('Mystem in files writing - {0}'.format(finish - start))

def run_mystem_subprocess(mystem_in_filename):
    flibusta_id = get_flibusta_id(mystem_in_filename)
    command = './mystem -cgid --format json --eng-gr {1}{0}.mystem.in {2}{0}.mystem.out'.format(
            flibusta_id,
            mystem_in_directory,
            mystem_out_directory
        )
    subprocess.call(command.split(' '))
    with logging_lock:
        add_log('{0} file mystemmed'.format(mystem_in_filename))

def init(logging_l):
    global logging_lock
    logging_lock = logging_l

def run_mystem(process_count):
    lock = multiprocessing.Lock()
    filenames = glob.glob('{0}*.mystem.in'.format(mystem_in_directory))
    pool = multiprocessing.Pool(process_count, initializer=init, initargs=(lock,))
    start = datetime.datetime.now()
    pool.map(run_mystem_subprocess, filenames) 
    finish = datetime.datetime.now()
    print('Mystemming - {0}'.format(finish - start))

def collect_features(mystem_out_filename):
    text_stats = AggregatedTextStats(get_flibusta_id(mystem_out_filename))
    with open(mystem_out_filename, 'r') as mystem_out:
        for paragraph in mystem_out:
            paragraph = paragraph[:-1]
            process_mystemed_paragraph(paragraph, text_stats)
    #with logging_lock:
    add_log('{0} mystem features collected'.format(mystem_out_filename))
    return text_stats
 
def collect_mystem_features(process_count):
    #lock = multiprocessing.Lock()
    filenames = glob.glob('{0}*.mystem.out'.format(mystem_out_directory))
    #pool = multiprocessing.Pool(process_count, initializer=init, initargs=(lock,))
    start = datetime.datetime.now()
    stats = []
    for filename in filenames:
        stats.append(collect_features(filename))
    #stats = pool.map(collect_features, filenames)
    finish = datetime.datetime.now()
    print('Collecting mystem features - {0}'.format(finish - start)) 
    with open('mystem_content_based_features.csv', 'w') as f:
        writer = csv.DictWriter(f, fieldnames=AggregatedTextStats.get_mystem_features_fieldnames())
        writer.writeheader()
        for mystem_stats in stats:
            writer.writerow(mystem_stats.get_mystem_features_row())

if __name__ == '__main__':
    get_mystem_in_files()
    run_mystem(2)
    # collect_mystem_features(5)
