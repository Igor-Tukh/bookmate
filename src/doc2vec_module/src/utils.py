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
        return True
    elif value == 'False':
        return False
    else:
        return None