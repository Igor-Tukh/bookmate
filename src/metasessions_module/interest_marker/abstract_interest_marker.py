class AbstractInterestMarker(object):
    @staticmethod
    def get_for_user(book_id, document_id, user_id):
        raise NotImplementedError

    @staticmethod
    def get_marker_title():
        raise NotImplementedError
