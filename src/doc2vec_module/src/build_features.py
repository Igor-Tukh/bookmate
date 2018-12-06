from src.doc2vec_module.src.process_epub import get_text_by_session


class FeaturesBuilder(object):
    def __init__(self, sessions):
        self.sessions = sessions
        self.features = []

    def build_features(self):
        for session in self.sessions:
            self.features.append(FeaturesBuilder.build_features_for_session(session))

    @staticmethod
    def build_features_for_session(session):
        text = get_text_by_session(session, session['book_id'], session['document_id'])
        return FeaturesBuilder.build_features_from_text(text)

    @staticmethod
    def build_features_from_text(text):
        pass
