class BookStats():
    def __init__(self):
        self._id = 0
        self.avr_sentence_len = 0.0
        self.symbols_num = 0
        self.dialogs_num = 0
        self.words_num = 0
        self.sentences_num = 0
        self.p_num = 0
        self.pages_num = 0
        self.avr_word_len = 0.0
        self.avr_dialogs_part = 0.0
        self.text = ''

    def to_dict(self):
        return self.__dict__