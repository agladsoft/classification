import re
import datetime


class ShipAndVoyage:

    @staticmethod
    def replace_symbols_and_letters(value, ocr_json_label):
        month_list = ["янв.", "февр.", "марта", "апр.", "мая", "июн.", "июл.", "авг.", "сент.", "окт.", "нояб.", "дек."]
        reg_exp = "\d{1,2}[^\S\n\t]+\w+.[^\S\n\t]+\d{4}"
        date = re.findall(reg_exp, value)[0].split()
        if date[1] in month_list:
            month_digit = month_list.index(date[1]) + 1
        date = datetime.datetime.strptime(f'{date[2]}-{str(month_digit)}-{date[0]}', "%Y-%m-%d")
        ocr_json_label["text"] = str(date.date())
        print(ocr_json_label["text"])
        # new_date = re.findall(reg_exp, value)[0].split()[0].replace("б", "6").replace("I", "1").replace("з", "3").replace("z", "2").replace("o", "0").replace("о", "0").replace("в", "8").replace("B", "8")
        # ocr_json_label["text"] = value.replace(date, new_date)

    @staticmethod
    def is_validate_for_ship_and_voyage(value, ocr_json_label):
        ocr_json_label["validation"] = bool(re.findall("\d{4}-\d{1,2}-\d{1,2}", value))


class ShipAndVoyage2:

    @staticmethod
    def replace_word(value, ocr_json_label):
        value = value.replace("на", "по")
        ocr_json_label["text"] = value

    @staticmethod
    def is_validate_for_ship_and_voyage2(value, ocr_json_label):
        ocr_json_label["validation"] = len(value.split()) == 3