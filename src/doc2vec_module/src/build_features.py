from src.doc2vec_module.src.process_epub import get_text_by_session, connect_to_mongo_database, \
    get_text_by_session_using_percents, get_epub_book_text_with_ebook_convert

import nltk
import string
import csv

BOOK_IDS = [('135089', '1222472')]
USER_IDS = {'1222472': ['42607', '374866', '1433804', '1540818', '1855471', '1970134', '1997078', '2067903', '2309986',
                        '2488778', '2497291', '2504830', '2558482', '2694654', '2738651', '2750150', '2810724']}

PUNCTUATION = string.punctuation


class FeaturesBuilder(object):
    def __init__(self, sessions, book_id):
        self.sessions = sessions
        self.book_id = book_id
        self.generalized_features = {}
        self.features = []

        self.text_features_builders = \
            {'words_number': FeaturesBuilder.calculate_words_number,
             'sentences_number': FeaturesBuilder.calculate_sentences_number,
             'words_number_normalized': FeaturesBuilder.calculate_words_number_normalized,
             'sentences_number_normalized': FeaturesBuilder.calculate_sentences_number_normalized,
             'average_word_len': FeaturesBuilder.calculate_average_word_len,
             'average_sentence_len': FeaturesBuilder.calculate_average_sentence_len}

        self.context_features_builders = \
            {'hour': FeaturesBuilder.get_session_hour}

        self.book_features_builders = \
            {'distance_from_the_beginning' : FeaturesBuilder.get_distance_from_the_beginning}

        self.text_features_list = [name for name, _ in self.text_features_builders.items()]
        self.context_features_list = [name for name, _ in self.context_features_builders.items()]
        self.book_features_list = [name for name, _ in self.book_features_builders.items()]

    def build_features(self,
                       certain_text_features=None,
                       certain_context_features=None,
                       certain_book_features=None,
                       save_features=True,
                       filename='features.csv',
                       clear_features=True,
                       text=None):
        if clear_features:
            self.features = []

        text_features = self.text_features_list if certain_text_features is None else certain_text_features
        context_features = self.context_features_list if certain_context_features is None else certain_context_features
        book_features = self.book_features_list if certain_book_features is None else \
            certain_book_features

        for session in self.sessions:
            session_features = self.build_text_features_for_session(session, book_text=text,
                                                                    certain_features_list=text_features)
            session_features.update(self.build_context_features_for_session(session,
                                                                            certain_context_features=context_features))
            session_features.update(self.build_book_features_for_session(session,
                                                                         book_text=text,
                                                                         certain_book_features=book_features))

            session_features['session_id'] = session['_id']
            session_features['speed'] = get_session_speed(session,
                                                          fix_old=True,
                                                          old_symbols_number=3568733,
                                                          current_symbols_number=len(text))
            self.features.append(session_features)

        if save_features:
            with open(filename, 'w') as csv_file:
                fields = text_features + context_features + book_features
                fields.append('session_id')
                fields.append('speed')
                writer = csv.DictWriter(csv_file, fieldnames=fields)
                writer.writeheader()

                for session_features in self.features:
                    writer.writerow(session_features)

        return self.features

    def upload_features(self, filename, clear_old=True):
        if clear_old:
            self.features = []

        with open(filename) as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                current_features = {}
                for field in reader.fieldnames:
                    current_features[field] = row[field]
                self.features.append(current_features)

    def get_features_by_names(self, feature_names):
        results = []
        for session_features in self.features:
            features = []
            for feature in feature_names:
                features.append(session_features[feature])
            results.append(features)
        return results

    def build_text_features_for_session(self, session, book_text=None, certain_features_list=None):
        # text = get_text_by_session(session, session['book_id'], session['document_id'])
        if book_text is None:
            text = get_text_by_session_using_percents(session, session['book_id'])
        else:
            text = get_text_by_session_using_percents(session, session['book_id'], book_text)

        words = nltk.word_tokenize(text)
        running_features = self.text_features_list if certain_features_list is None else certain_features_list
        current_features = {}
        for feature in running_features:
            current_features[feature] = self.text_features_builders[feature](text, words)

        return current_features

    def build_context_features_for_session(self, session, certain_context_features=None):
        feature_list = self.context_features_list if certain_context_features is None else certain_context_features
        current_features = {}
        for feature in feature_list:
            current_features[feature] = self.context_features_builders[feature](session)

        return current_features

    def build_book_features_for_session(self, session, book_text=None, certain_book_features=None):
        if book_text is None:
            text = get_text_by_session_using_percents(session, session['book_id'])
        else:
            text = get_text_by_session_using_percents(session, session['book_id'], book_text)

        feature_list = self.context_features_list if certain_book_features is None else certain_book_features
        current_features = {}
        for feature in feature_list:
            current_features[feature] = self.book_features_builders[feature](session, self.book_id, text)

        return current_features

    @staticmethod
    def calculate_words_number(text, words):
        count = 0
        for word in words:
            if word not in PUNCTUATION:
                count += 1
        return count

    @staticmethod
    def calculate_sentences_number(text, words):
        count = 1
        for word in words:
            if word == '.':
                count += 1
        return count

    @staticmethod
    def calculate_words_number_normalized(text, words):
        return FeaturesBuilder.calculate_words_number(text, words) / len(text)

    @staticmethod
    def calculate_sentences_number_normalized(text, words):
        return FeaturesBuilder.calculate_sentences_number(text, words) / len(words)

    @staticmethod
    def calculate_average_word_len(text, words):
        if len(words) == 0:
            return 0
        return sum([len(word) for word in words]) / len(words)

    @staticmethod
    def calculate_average_sentence_len(text, words):
        total_len = int(FeaturesBuilder.calculate_average_word_len(text, words) * len(words))
        sentensece_number = FeaturesBuilder.calculate_sentences_number(text, words)
        if sentensece_number == 1:
            return 0
        return total_len / FeaturesBuilder.calculate_sentences_number(text, words)

    @staticmethod
    def get_session_hour(session):
        return session['read_at'].hour

    @staticmethod
    def get_distance_from_the_beginning(session, book_id, text):
        return session['book_from'] / 100


def get_user_sessions(book_id, user_id):
    db = connect_to_mongo_database('bookmate')
    return db[book_id].find({'user_id': int(user_id)})


def get_session_speed(session, fix_old=False, old_symbols_number=0, current_symbols_number=0):
    if not fix_old:
        return session['speed']
    return (session['speed'] / old_symbols_number) * current_symbols_number


if __name__ == '__main__':
    for book_id, document_id in BOOK_IDS:
        text = get_epub_book_text_with_ebook_convert(book_id)
        for user_id in USER_IDS[document_id]:
            builder = FeaturesBuilder(get_user_sessions(book_id, user_id), book_id)
            builder.build_features(filename='../features/{user_id}.csv'.format(user_id=user_id), text=text)
