class PageStats(object):
    def __init__(self):
        self._id = -1
        self.person_verbs_num = 0
        self.person_verbs_part = 0.0
        self.dialogs_part = 0.0
        self.dialogs_num = 0
        self.person_pronouns_num = 0
        self.person_pronouns_part = 0.0
        self.words_num = 0
        self.avr_word_len = 0.0
        self.text = ''
        self.sentences_num = 0
        self.symbols_num = 0
        self.end_of_section = False
        self.begin_of_section = False
        self.new_words_count = 0
        self.sentiment_word_part = 0.0
        self.labeled_word_part = 0.0
        self.labeled_word_num = 0
        self.begin_symbol_pos = 0
        self.p_num = 0
        self._to = 0
        self._from = 0
        self.clear_text = ''
        self.section_num = 0

    def to_dict(self):
        return self.__dict__


