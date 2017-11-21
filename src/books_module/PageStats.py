class PageStats(object):
    def __init__(self):
        # textual features
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

        self.new_words_count = 0
        # self.labeled_word_part = 0.0
        # self.labeled_word_num = 0
        # self.begin_symbol_pos = 0
        self.p_num = 0
        self.symbol_to = 0
        self.symbol_from = 0
        self.clear_text = ''

        # behaviour features
        self.page_speed = 0
        self.page_sessions = 0
        self.page_skip_percent = 0
        self.page_unusual_percent = 0
        self.page_return_percent = 0


    def to_dict(self):
        return self.__dict__


