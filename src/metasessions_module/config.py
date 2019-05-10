from enum import Enum

BOOKS = {'The Fault in Our Stars': 266700, 'Fifty Shades of Grey': 210901}
DOCUMENTS = {210901: [1143157, 1416430, 1311858], 266700: [969292, 776328, 823395]}

BOOK_LABELS = {210901: ['Знакомство', 'Покупки Грея', 'Личное общение', 'Пьяная, домогательства',
                        'Планы', 'Подпись бумаг', 'Комната для игр', 'Секс', 'Завтрак, секс',
                        'Признание', 'Контракт', 'Отказ, секс', 'Обсуждение контракта',
                        'Вручение диплома', 'Обсуждение, секс', 'Порка, секс', 'Хлопотоы',
                        'Гинеколог, секс', 'Ужин', 'Секс, детство', 'Секс, собеседования',
                        'Перелет, ссора', 'Секс, секс', 'Планеризм', 'Секс', 'Наказание, все кончено'],
               266700: ['Группа поддержки', 'История Хейзел (рак)', 'Общение угнетает',
                        'Роман, дальнеяшая судьба', 'Желание -- в Голландию', 'Поцелуй, она - граната',
                        'Страшные боли, все равно в Голландию?', 'В Голландию можно!', 'В преддверии поездки',
                        'Признание в любви', 'Романтический ужин', 'Встреча с писателем, слезы, секс',
                        'Метастазы, конец ремиссии', 'Поддержка, закидывание яйцами',
                        'Реанимация', 'Последняя стадия', 'Не будет некролога во всех газетах',
                        'Случай на бензозаправке', 'Лишен иллюзий окончательно и безоговорочно',
                        'Последний хороший день', 'Гас умер', 'Панихида, разговор с писателем',
                        'Смерть дочери писателя от рака', 'Родители после смерти?', 'Любовь -- тот самый след']}


INFINITE_SPEED = 10000000
UNKNOWN_SPEED = -1


class ReadingStyle(Enum):
    SCANNING = 1
    SKIMMING = 2
    NORMAL = 3
    DETAILED = 4

    def get_color(self):
        if self == ReadingStyle.SCANNING:
            return 'ro'
        elif self == ReadingStyle.SKIMMING:
            return 'go'
        elif self == ReadingStyle.NORMAL:
            return 'bo'
        elif self == ReadingStyle.DETAILED:
            return 'yo'

    @staticmethod
    def get_reading_style_by_speed(speed,
                                   scanning_lower_thrshold=3500.0,
                                   skimming_lower_threshold=2500,
                                   normal_lower_threshold=1150):
        if speed > scanning_lower_thrshold:
            return ReadingStyle.SCANNING
        elif speed > skimming_lower_threshold:
            return ReadingStyle.SKIMMING
        elif speed > normal_lower_threshold:
            return ReadingStyle.NORMAL
        return ReadingStyle.DETAILED
