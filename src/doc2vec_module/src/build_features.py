from src.doc2vec_module.src.process_epub import get_text_by_session

import nltk
import string
import csv


USER_IDS = ['42607', '374866', '1433804', '1540818', '1855471', '1970134', '1997078', '2067903', '2309986',
            '2488778', '2497291', '2504830', '2558482', '2694654', '2738651', '2750150', '2810724']

PUNCTUATION = string.punctuation


class FeaturesBuilder(object):
    def __init__(self, sessions):
        self.sessions = sessions
        self.features = []
        self.text_features_builders = {'words_number': FeaturesBuilder.calculate_words_number,
                                       'sentences_number': FeaturesBuilder.calculate_sentences_number,
                                       'average_word_len': FeaturesBuilder.calculate_average_word_len,
                                       'average_sentence_len': FeaturesBuilder.calculate_average_sentence_len}

        self.text_features_list = [name for name, _ in self.text_features_builders.items()]

    def build_features(self, certain_text_features=None, save_features=True, filename='features', clear_features=True):
        if clear_features:
            self.features = []

        text_features = self.text_features_list if certain_text_features is None else certain_text_features
        for session in self.sessions:
            session_features = self.build_text_features_for_session(session)
            session_features['session_id'] = session['_id']
            self.features.append(session_features)
            # for other session_features.update()

        if not save_features:
            return

        with open(filename + '.csv', 'w') as csv_file:
            fields = text_features  # to other + ...
            fields.append('session_id')
            writer = csv.DictWriter(csv_file, fieldnames=fields)
            writer.writeheader()

            for session_features in self.features:
                writer.writerow(session_features)

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

    def build_text_features_for_session(self, session, certain_features_list=None):
        text = get_text_by_session(session, session['book_id'], session['document_id'])
        words = nltk.word_tokenize(text)
        running_features = self.text_features_list if certain_features_list is None else certain_features_list
        current_features = {}
        for feature in running_features:
            current_features[feature] = self.text_features_builders[feature](words)
        return current_features

    @staticmethod
    def calculate_words_number(words):
        count = 0
        for word in words:
            if word not in PUNCTUATION:
                count += 1
        return count

    @staticmethod
    def calculate_sentences_number(words):
        count = 0
        for word in words:
            if word == '.':
                count += 1
        return count

    @staticmethod
    def calculate_average_word_len(words):
        return sum([len(word) for word in words]) / len(words)

    @staticmethod
    def calculate_average_sentence_len(words):
        total_len = int(FeaturesBuilder.calculate_average_word_len(words) * len(words))
        return total_len / FeaturesBuilder.calculate_sentences_number(words)


if __name__ == '__main__':
    pass
