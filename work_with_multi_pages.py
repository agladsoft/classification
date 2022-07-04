import os
import re
import sys
from PIL import Image
from scipy.ndimage import interpolation as inter
import math
import logging
import subprocess
import shutil
import cv2
import numpy as np
import pytesseract
from PyPDF2 import PdfFileReader
from MyJSONFormatter import MyJSONFormatter
import yaml


class WorkMultiPages:
    def __init__(self, input_file, cache, output_path):
        self.input_file = input_file
        self.cache = cache
        self.output_path = output_path

    @staticmethod
    def logging_init(filename, get_logger_out, get_logger):
        data_json = 'data_json'
        if not os.path.exists(data_json):
            os.makedirs(f'{data_json}')

        formatter = MyJSONFormatter()
        console_out = logging.StreamHandler()
        console_out.setFormatter(formatter)
        logger_out = logging.getLogger(get_logger_out)
        logger_out.addHandler(console_out)
        logger_out.setLevel(logging.INFO)

        json_handler = logging.FileHandler(filename=f'{data_json}/{filename}.json')
        json_handler.setFormatter(formatter)
        logger = logging.getLogger(get_logger)
        logger.addHandler(json_handler)
        logger.setLevel(logging.INFO)
        return logger, logger_out

    def len_files_in_file(self):
        logger, logger_out = self.logging_init('files_in_file', 'len_files_in_file_out', 'len_files_in_file')
        with open(self.input_file, 'rb') as fl:
            reader = PdfFileReader(fl)
            count_pages = reader.getNumPages()
        logger.info(f'{os.path.basename(self.input_file)}', extra={f'count_files': count_pages})
        logger_out.info(f"Количество файлов pdf в многостраничном файле: {count_pages}")

    def split_files_in_file(self):
        result = subprocess.run(["pdfimages", "-all", self.input_file, self.cache], stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, text=True)
        return result

    def len_files_in_cache(self):
        logger, logger_out = self.logging_init('files_in_cache', 'len_files_in-cache_out', 'len_files_in-cache')
        path, dirs, files = next(os.walk(f"{os.path.dirname(self.cache)}"))
        file_count = len(files)
        logger.info(f'{os.path.basename(self.input_file)}', extra={'count_files': file_count})
        logger_out.info(f"Количество разбитых файлов pdf в кэше: {file_count}")
        return file_count

    @staticmethod
    def rotate(image, angle, background):
        old_width, old_height = image.shape[:2]
        angle_radian = math.radians(angle)
        width = abs(np.sin(angle_radian) * old_height) + abs(np.cos(angle_radian) * old_width)
        height = abs(np.sin(angle_radian) * old_width) + abs(np.cos(angle_radian) * old_height)

        image_center = tuple(np.array(image.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
        rot_mat[1, 2] += (width - old_width) / 2
        rot_mat[0, 2] += (height - old_height) / 2
        return cv2.warpAffine(image, rot_mat, (int(round(height)), int(round(width))), borderValue=background)

    def turn_img(self):
        logger, logger_out = self.logging_init('turn_img', 'turn_img_out', 'turn_img')
        if not os.path.exists("all_files/dir_classific"):
            # os.makedirs(os.path.join("all_files/dir_classific", "garbage"))
            os.makedirs(os.path.join("all_files/dir_classific", "line"))
            os.makedirs(os.path.join("all_files/dir_classific", "port"))
            # os.makedirs(os.path.join("all_files/dir_classific", "two_page_port"))

            os.makedirs(os.path.join("all_files/dir_classific", "contract"))
            os.makedirs(os.path.join("all_files/dir_classific", "unknown"))

        im = cv2.imread(str(self.input_file))
        rotate_img = pytesseract.image_to_osd(im)
        angle_rotated_image = int(re.search('(?<=Orientation in degrees: )\d+', rotate_img).group(0))
        rotated = self.rotate(im, angle_rotated_image, (0, 0, 0))
        file_name = os.path.basename(self.input_file)
        cv2.imwrite(f'{os.path.dirname(self.cache)}/{file_name}', rotated)
        logger.info(f'{os.path.basename(file_name)}', extra={'rotate': angle_rotated_image})
        logger_out.info(f'Rotate: {angle_rotated_image}, Filename: {os.path.basename(file_name)}')

    def correct_skew(self, delta, limit):
        def determine_score(arr, angle):
            data = inter.rotate(arr, angle, reshape=False, order=0)
            histogram = np.sum(data, axis=1, dtype=float)
            score = np.sum((histogram[1:] - histogram[:-1]) ** 2, dtype=float)
            return histogram, score

        logger, logger_out = self.logging_init('turn_small_img', 'turn_small_img_out', 'turn_small_img')
        image = cv2.imread(self.input_file)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

        scores = []
        angles = np.arange(-limit, limit + delta, delta)
        for angle in angles:
            histogram, score = determine_score(thresh, angle)
            scores.append(score)

        best_angle = angles[scores.index(max(scores))]

        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, best_angle, 1.0)
        corrected = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        logger.info(f'{os.path.basename(self.input_file)}', extra={'skew': best_angle})
        logger_out.info(f'Skew is: {best_angle:.04f}, Filename: {os.path.basename(self.input_file)}')
        cv2.imwrite(f"{os.path.dirname(self.cache)}/{os.path.basename(self.input_file)}", corrected)

    @staticmethod
    def normalize_box(box, width, height):
        return [
            int(1000 * (box[0] / width)),
            int(1000 * (box[1] / height)),
            int(1000 * (box[2] / width)),
            int(1000 * (box[3] / height)),
        ]

    @staticmethod
    def move_file_in_dir(str_of_doc, file_name, predict=None):
        with open("config_yaml/classification/classification.yml", "r") as stream:
            try:
                yaml_file = yaml.safe_load(stream)
                for len_label_in_config in range(len(yaml_file["classification"])):
                    if yaml_file["classification"][len_label_in_config]["key"] in str_of_doc:
                        shutil.move(file_name, yaml_file["classification"][len_label_in_config]['folder'])
                        predict = yaml_file["classification"][len_label_in_config]['name']

                if not predict:
                    shutil.move(file_name, yaml_file["classification"][-1]['folder'])
                    predict = yaml_file["classification"][-1]['name']
            except yaml.YAMLError as exc:
                print(exc)
        return predict

    def classification_img(self):
        logger, logger_out = self.logging_init('predict_img', 'predict_img_out', 'predict_img')

        image = Image.open(self.input_file)
        ocr_df = pytesseract.image_to_data(image, output_type='data.frame', lang='eng+rus')
        float_cols = ocr_df.select_dtypes('float').columns
        ocr_df[float_cols] = ocr_df[float_cols].round(0).astype(int)
        ocr_df = ocr_df.replace(r'^\s*$', np.nan, regex=True)
        ocr_df = ocr_df.dropna().reset_index(drop=True)
        words = list(ocr_df.text)
        str_of_doc = " ".join(words[:20])
        predict = self.move_file_in_dir(str_of_doc, self.input_file)
        logger_out.info(f'Filename: {os.path.basename(self.input_file)}, Predict class: {predict}')
        logger.info(f'{os.path.basename(self.input_file)}', extra={'file_name': os.path.basename(self.input_file),
                                                                 'predict': predict, "text": str_of_doc})
        return predict

    def __call__(self, *args, **kwargs):
        # self.len_files_in_file()
        # self.split_files_in_file()
        # self.len_files_in_cache()
        self.turn_img()
        self.correct_skew(delta=1, limit=15)
        return self.classification_img()


if __name__ == "__main__":
    work_multi_pages = WorkMultiPages(sys.argv[1], sys.argv[2], sys.argv[3])
    print(work_multi_pages())


