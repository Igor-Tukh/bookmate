import time
from datetime import datetime


def is_float(value):
    try:
        float(value)
        return True
    except ValueError:
        return False


def is_str_of_bool(value):
    return value in {'True', 'False'}


def str_to_bool(value):
    if value == 'True':
        return 1.0
    elif value == 'False':
        return 0.0
    else:
        return None


def str_to_timestamp(string_value):
    date_time = str_to_datetime(string_value)
    if date_time is None:
        return None
    return time.mktime(date_time.timetuple())


def str_to_datetime(string_value):
    try:
        val = datetime.strptime(string_value, '%Y-%m-%d %H:%M:%S')
        return val
    except:
        return None
