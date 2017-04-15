import datetime

def date_from_timestamp(timestamp):
    return datetime.datetime.fromtimestamp(timestamp//1000)

def percent_field_to_float(percent):
    if type(percent) is float:
        return percent
    if type(percent) is str:
        try:
            return float(percent)
        except:
            pass
    return None

class Record(object):
    def __init__(self, db_record, lcid_bid):
        if db_record is None:
            self.empty = True
            return
        self.empty = False
        self.library_card_id = db_record[0]
        self.read_at = date_from_timestamp(db_record[1])
        self.is_uploaded = (db_record[2] == 1)
        self.app_user_agent = db_record[3]
        self.user_id = db_record[4]
        self.ip = db_record[5]
        self.created_at = date_from_timestamp(db_record[6])
        self.phantom_id = db_record[7]
        self.updated_at = date_from_timestamp(db_record[8])
        self.author_page = db_record[9]
        self.country_code3 = db_record[10]
        self.to_percent = percent_field_to_float(db_record[11])
        self.user_agent = db_record[12]
        self.book_id = db_record[13]
        if self.book_id is None and self.library_card_id is not None and \
                str(self.library_card_id) in lcid_bid and lcid_bid[str(self.library_card_id)].isdigit():
            self.book_id = int(lcid_bid[str(self.library_card_id)])
        self.item_id = db_record[14]
        self.from_percent = percent_field_to_float(db_record[15])
        self.record_id = db_record[16]
        self.is_phantom = (db_record[17] == 1)
        self.city = db_record[18]
        self.document_id = db_record[19]
        self.size = db_record[20]

    # def __init__(self, db_record, lcid_bid):
    #     if db_record is None:
    #         self.empty = True
    #         return
    #     self.user_id = db_record[0]
    #     self.book_id = db_record[1]
    #     self.from_percent = 0
    #     self.to_percent = 0
    #     self.read_at = 0
    #     self.empty = False
