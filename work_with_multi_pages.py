import glob
import os
import re
import sys
from PIL import Image
# import torch
# from transformers import LayoutLMTokenizer
from scipy.ndimage import interpolation as inter
import math
import logging
import subprocess
import shutil
# import statistics
import cv2
import numpy as np
import pytesseract
# from scipy import ndimage
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

        for file_name in sorted(glob.glob(f"{os.path.dirname(self.cache)}/*.jpg")):
            im = cv2.imread(str(file_name))
            rotate_img = pytesseract.image_to_osd(im)
            angle_rotated_image = int(re.search('(?<=Orientation in degrees: )\d+', rotate_img).group(0))
            rotated = self.rotate(im, angle_rotated_image, (0, 0, 0))
            # if angle_rotated_image > 0 and angle_rotated_image != 180:
            #     shutil.move(file_name, f"all_files/dir_classific/line")
            # else:
            file_name = os.path.basename(file_name)
            cv2.imwrite(f'{os.path.dirname(self.cache)}/{file_name}', rotated)
            logger.info(f'{os.path.basename(file_name)}', extra={'rotate': angle_rotated_image})
            logger_out.info(f'Rotate: {angle_rotated_image}, Filename: {os.path.basename(file_name)}')

    # def turn_img_small_degree(self):
    #     logger, logger_out = self.logging_init('turn_small_img', 'turn_small_img_out', 'turn_small_img')
    #     for file_full_name in sorted(glob.glob(f"{os.path.dirname(self.cache)}/*.jpg")):
    #         foo = Image.open(file_full_name)
    #         (width, height) = foo.size
    #         foo = foo.resize((width // 4, height // 4), Image.ANTIALIAS)
    #         foo.save("resized_img.jpg", optimize=True, quality=95)
    #
    #         img_for_define_angle = cv2.imread("resized_img.jpg")
    #         img_to_save = cv2.imread(file_full_name)
    #         img_gray = cv2.cvtColor(img_for_define_angle, cv2.COLOR_BGR2GRAY)
    #         img_edges = cv2.Canny(img_gray, 100, 100, apertureSize=3)
    #         lines = cv2.HoughLinesP(img_edges, 1, math.pi / 180.0, 100, minLineLength=100, maxLineGap=5)
    #         angles = []
    #         for [[x1, y1, x2, y2]] in lines:
    #             angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
    #             angles.append(angle)
    #         median_angle = statistics.median(angles)
    #         logger.info(
    #             f'{os.path.basename(file_full_name)}',
    #             extra={'rotate': median_angle},
    #         )
    #         logger_out.info(f'Angle is: {median_angle:.04f}, Filename: {os.path.basename(file_full_name)}')
    #         if (15 > median_angle > 0) or (-15 < median_angle < 0):
    #             img_rotated = ndimage.rotate(img_to_save, median_angle)
    #             file_name = os.path.basename(file_full_name)
    #             cv2.imwrite(f"{os.path.dirname(self.cache)}/{file_name}", img_rotated)
    #     os.remove("resized_img.jpg")

    def correct_skew(self, delta, limit):
        def determine_score(arr, angle):
            data = inter.rotate(arr, angle, reshape=False, order=0)
            histogram = np.sum(data, axis=1, dtype=float)
            score = np.sum((histogram[1:] - histogram[:-1]) ** 2, dtype=float)
            return histogram, score

        logger, logger_out = self.logging_init('turn_small_img', 'turn_small_img_out', 'turn_small_img')
        for full_image in sorted(glob.glob(f"{os.path.dirname(self.cache)}/*.jpg")):
            image = cv2.imread(full_image)
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
            logger.info(f'{os.path.basename(full_image)}', extra={'skew': best_angle})
            logger_out.info(f'Skew is: {best_angle:.04f}, Filename: {os.path.basename(full_image)}')
            cv2.imwrite(f"{os.path.dirname(self.cache)}/{os.path.basename(full_image)}", corrected)

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
        with open("config_yaml/classification.yml", "r") as stream:
            try:
                yaml_file = yaml.safe_load(stream)
                for len_label_in_config in range(len(yaml_file["classification"])):
                    if yaml_file["classification"][len_label_in_config]["key"] in str_of_doc:
                        shutil.move(file_name, yaml_file["classification"][len_label_in_config]['folder'])
                        predict = yaml_file["classification"][len_label_in_config]['name']
                        # logger_out.info(f'Filename: {os.path.basename(file_name)}, Predict class: {predict}')
                        # logger.info(f'{os.path.basename(file_name)}', extra={'file_name': os.path.basename(
                        #     file_name), 'predict': predict, "text": str_of_doc})

                if not predict:
                    shutil.move(file_name, yaml_file["classification"][-1]['folder'])
                    predict = yaml_file["classification"][-1]['name']

                        # for key in yaml_file:
                        #     if yaml_file[key]['-key'] in str_of_doc:
                        #         shutil.move(file_name, yaml_file[key]['-folder'])
                        #         predict = yaml_file[key]['-name']
                        #         return yaml_file[key]['-config']
                        # try:
                        #     predict
                        # except UnboundLocalError:
                        #     shutil.move(file_name, yaml_file["unknown"]['-folder'])
                        #     predict = yaml_file["unknown"]['-name']
            except yaml.YAMLError as exc:
                print(exc)
        return predict

    def classification_img(self, model_path):
        logger, logger_out = self.logging_init('predict_img', 'predict_img_out', 'predict_img')

        for file_name in sorted(glob.glob(f"{os.path.dirname(self.cache)}/*.jpg")):
            image = Image.open(file_name)
            width, height = image.size
            # apply ocr to the image
            ocr_df = pytesseract.image_to_data(image, output_type='data.frame', lang='eng+rus')
            float_cols = ocr_df.select_dtypes('float').columns
            ocr_df[float_cols] = ocr_df[float_cols].round(0).astype(int)
            ocr_df = ocr_df.replace(r'^\s*$', np.nan, regex=True)
            ocr_df = ocr_df.dropna().reset_index(drop=True)
            words = list(ocr_df.text)
            str_of_doc = " ".join(words[:20])
            predict = self.move_file_in_dir(str_of_doc, file_name)
            # if 'Поручение на погрузку' in str_of_doc:
            #     shutil.move(file_name, "all_files/dir_classific/port")
            #     predict = 'port'
            # else:
            #     shutil.move(file_name, "all_files/dir_classific/line")
            #     predict = 'line'

            logger_out.info(f'Filename: {os.path.basename(file_name)}, Predict class: {predict}')
            logger.info(f'{os.path.basename(file_name)}', extra={'file_name': os.path.basename(file_name),
                                                                 'predict': predict, "text": str_of_doc})

        # PATH = model_path
        # model = torch.load(PATH)
        # tokenizer = LayoutLMTokenizer.from_pretrained('microsoft/layoutlm-base-uncased')
        # idx2label = {0: "two_page_port", 1: "garbage", 2: "line", 3: "port"}
        # for file_name in sorted(glob.glob(f"{os.path.dirname(self.cache)}/*.jpg")):
        #     image = Image.open(file_name)
        #     width, height = image.size
        #     # apply ocr to the image
        #     ocr_df = pytesseract.image_to_data(image, output_type='data.frame', lang='eng+rus')
        #     float_cols = ocr_df.select_dtypes('float').columns
        #     ocr_df[float_cols] = ocr_df[float_cols].round(0).astype(int)
        #     ocr_df = ocr_df.replace(r'^\s*$', np.nan, regex=True)
        #     ocr_df = ocr_df.dropna().reset_index(drop=True)
        #     words = list(ocr_df.text)
        #     words = words[:50]
        #     coordinates = ocr_df[['left', 'top', 'width', 'height']]
        #     actual_boxes = []
        #     for idx, row in coordinates.iterrows():
        #         x, y, w, h = tuple(row)  # the row comes in (left, top, width, height) format
        #         actual_box = [x, y, x + w,
        #                       y + h]  # we turn it into (left, top, left+width, top+height) to get the actual box
        #         actual_boxes.append(actual_box)
        #
        #     # normalize the bounding boxes
        #     boxes = []
        #     for box in actual_boxes:
        #         boxes.append(self.normalize_box(box, width, height))
        #
        #     token_boxes = []
        #     for word, box in zip(words, boxes):
        #         word_tokens = tokenizer.tokenize(word)
        #         token_boxes.extend([box] * len(word_tokens))
        #     # add bounding boxes of cls + sep tokens
        #     token_boxes = [[0, 0, 0, 0]] + token_boxes + [[1000, 1000, 1000, 1000]]
        #
        #     encoding = tokenizer(' '.join(words), return_tensors="pt")
        #     input_ids = encoding["input_ids"]
        #     attention_mask = encoding["attention_mask"]
        #     token_type_ids = encoding["token_type_ids"]
        #     bbox = torch.tensor([token_boxes])
        #     sequence_label = torch.tensor([1])
        #     outputs = model(input_ids=input_ids, bbox=bbox, attention_mask=attention_mask,
        #                     token_type_ids=token_type_ids,
        #                     labels=sequence_label)
        #     loss = outputs.loss
        #     logits = outputs.logits
        #     dict_predict = dict()
        #     predict_list = [x.item() for x in logits[0]]
        #     for name, predict in zip(idx2label.values(), predict_list):
        #         dict_predict[name] = round(predict, 2)
        #     predicted_class_idx = logits.argmax(-1).item()
        #     predict = idx2label[predicted_class_idx]
        #     logger_out.info(
        #         f'Filename: {os.path.basename(file_name)}, Predict class: {predict}, Likelihood: {dict_predict}')
        #     logger.info(f'{os.path.basename(file_name)}',
        #                 extra={'file_name': os.path.basename(file_name), 'predict': predict, **dict_predict,
        #                        "text": words})
        #
        #     if predict == 'line':
        #         shutil.move(file_name, "all_files/dir_classific/line")
        #     elif predict == 'port':
        #         shutil.move(file_name, "all_files/dir_classific/port")
        #     elif predict == 'two_page_port':
        #         shutil.move(file_name, "all_files/dir_classific/two_page_port")
        #     elif predict == 'garbage':
        #         shutil.move(file_name, "all_files/dir_classific/garbage")

    def __call__(self, *args, **kwargs):
        self.len_files_in_file()
        self.split_files_in_file()
        self.len_files_in_cache()
        self.turn_img()
        # self.turn_img_small_degree()
        self.correct_skew(delta=1, limit=15)
        self.classification_img('another_model_for_classification_documents_40_files_100_epoch.pth')


if __name__ == "__main__":
    work_multi_pages = WorkMultiPages(sys.argv[1], sys.argv[2], sys.argv[3])
    # work_multi_pages = WorkMultiPages(
    #     '/home/timur/PycharmWork/classification_text/all_files/7579  OAKLAND от 05.12.2021_1.pdf',
    #     '/home/timur/PycharmWork/classification_text/cache/7579  OAKLAND от 05.12.2021_1.pdf',
    #     '/home/timur/PycharmWork/classification_text/dir_classific'
    # )
    work_multi_pages()


