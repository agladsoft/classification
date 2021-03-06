import itertools
import json
import xml.etree.ElementTree as ET
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import yaml
import sys
import re
import psycopg2

try:
    from PIL import Image
except ImportError:
    import Image
import pytesseract

file = sys.argv[1]
output_directory = sys.argv[2]
output_directory_csv = sys.argv[3]
config_yaml_file = sys.argv[4]
scripts_for_validations_and_postprocessing = sys.argv[5].replace('/', '.').replace(".py", "")

img = cv2.imread(file, 0)
predicted_boxes_dict = dict()
predection_box = list()
ocr_json = dict()
ocr_json_label_main = dict()
ocr_json_label = dict()
ship_voyage = []

data = pytesseract.image_to_data(img, output_type='dict', lang='rus+eng')
boxes = len(data['level'])
list_i = list()
dict_text = dict()
list_score = list()
text_ocr = ''
for i in range(boxes):
    (x, y, w, h) = (data["left"][i], data["top"][i], data["width"][i], data["height"][i])
    if data["text"][i] and not list_i[-1]:
        text_ocr = str(data["text"][i])
        x_min = x
        y_min = y
        list_score.clear()
        list_score.append(data["conf"][i])
    if data["text"][i] and list_i[-1]:
        text_ocr += " " + str(data["text"][i])
        list_score.append(data["conf"][i])
        x_max = x + w
        y_max = y + h
    elif not data["text"][i]:
        try:
            dict_text[text_ocr] = (x_min, y_min, x_max, y_max, np.mean(list_score), np.std(list_score))
        except:
            pass
    list_i.append(data["text"][i])

with open(config_yaml_file, "r") as stream:
    try:
        yaml_file = yaml.safe_load(stream)
        label_list = list()
        is_validation = None
        try:
            del_cell = yaml_file['config_of_table']['del_cell']
            length_of_kernel = yaml_file['config_of_table']['length_of_kernel']
            count_of_tables = yaml_file['config_of_table']['count_of_tables']
            min_width_of_cell = yaml_file['config_of_table']['min_width_of_cell']
            min_height_of_cell = yaml_file['config_of_table']['min_height_of_cell']
            indent_x_text_of_cells = yaml_file['config_of_table']['indent_x_text_of_cells']
            indent_y_text_of_cells = yaml_file['config_of_table']['indent_y_text_of_cells']
            config_for_pytesseract = yaml_file['config_of_table']['config_for_pytesseract']
        except KeyError as ex_key_error:
            pass

        try:
            database = yaml_file['config_of_database']['database']
            user = yaml_file['config_of_database']['user']
            password = yaml_file['config_of_database']['password']
            host = yaml_file['config_of_database']['host']
            port = yaml_file['config_of_database']['port']
            table = yaml_file['config_of_database']['table']
        except KeyError as ex_key_error:
            pass

        for value in list(dict_text.keys()):
            ocr_json_label = dict()
            for len_label_in_config in range(len(yaml_file["labels"])):
                if yaml_file["labels"][len_label_in_config]["key"] in value:
                    ship_voyage.append(value)
                    ocr_json_label["text"] = value
                    ocr_json_label["label"] = yaml_file["labels"][len_label_in_config]["label"]
                    ocr_json_label["xmin"] = dict_text[value][0]
                    ocr_json_label["ymin"] = dict_text[value][1]
                    ocr_json_label["xmax"] = dict_text[value][2]
                    ocr_json_label["ymax"] = dict_text[value][3]
                    ocr_json_label["score"] = dict_text[value][4]
                    ocr_json_label["std"] = dict_text[value][5]

                    try:
                        # postprocessing = locals()[yaml_file["labels"][len_label_in_config]["postprocessing"]]
                        class_name = yaml_file["labels"][len_label_in_config]["label"]
                        method_name = yaml_file["labels"][len_label_in_config]["postprocessing"]
                        imported = __import__(scripts_for_validations_and_postprocessing, fromlist=["*"])
                        class_name = getattr(imported, class_name)
                        postprocessing = getattr(class_name, method_name)
                        postprocessing(value, ocr_json_label)

                        # postprocessing = eval(class_name + '.' + method_name)
                        # postprocessing(value, ocr_json_label)
                    except KeyError as ex_key:
                        print("Not found the key by name postprocessing")

                    try:
                        class_name = yaml_file["labels"][len_label_in_config]["label"]
                        method_name = yaml_file["labels"][len_label_in_config]["validations"]
                        imported = __import__(scripts_for_validations_and_postprocessing, fromlist=["*"])
                        class_name = getattr(imported, class_name)
                        validations = getattr(class_name, method_name)
                        validations(ocr_json_label["text"], ocr_json_label)

                        # validations = eval(class_name + '.' + method_name)
                        # validations(ocr_json_label["text"], ocr_json_label)
                    except KeyError:
                        ocr_json_label["validation"] = True if ocr_json_label["score"] > 85 else False

                    label_list.append(ocr_json_label)
        ocr_json_label_main["type"] = "label"
        ocr_json_label_main["cells"] = label_list
        predection_box.append(ocr_json_label_main)
        table_list = list()
        outer = []
    except yaml.YAMLError as exc:
        print(exc)
    except TypeError as ex_type_error:
        img = cv2.imread(file, 1)
        noiseless_image_colored = cv2.fastNlMeansDenoisingColored(img, None, 20, 20, 7, 21)
        data = pytesseract.image_to_string(noiseless_image_colored, lang='rus+eng')

        # if "unknown.yml" in config_yaml_file:
        #     with open(output_directory + '/' + os.path.basename(file + '.txt'), "w") as f:
        #         f.writelines(data)
        # else:
        ocr_json_label["text"] = data
        ocr_json_label["label"] = yaml_file["labels"][len_label_in_config]["label"]
        label_list.append(ocr_json_label)

        ocr_json_label_main["type"] = "label"
        ocr_json_label_main["cells"] = label_list
        predection_box.append(ocr_json_label_main)
        table_list = list()
        outer = []

try:
    # ocr_json_label_main["type"] = "label"
    # ocr_json_label_main["cells"] = label_list
    # predection_box.append(ocr_json_label_main)
    # table_list = list()
    # outer = []
    # thresholding the image to a binary image
    thresh, img_bin = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # thresh, img_bin = cv2.threshold(img, 128, 255, cv2.THRESH_TOZERO | cv2.THRESH_OTSU)
    # inverting the image
    img_bin = 255 - img_bin

    # cv2.imwrite('cv_inverted.png', img_bin)
    # Plotting the image to see the output
    plotting = plt.imshow(img_bin, cmap='gray')
    # plt.show()

    # Length(width) of kernel as 100th of total width
    kernel_len = np.array(img).shape[1] // length_of_kernel
    # Defining a vertical kernel to detect all vertical lines of image
    ver_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_len))
    # Defining a horizontal kernel to detect all horizontal lines of image
    hor_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_len, 1))
    # A kernel of 2x2
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))

    # Use vertical kernel to detect and save the vertical lines in a jpg
    image_1 = cv2.erode(img_bin, ver_kernel, iterations=3)
    vertical_lines = cv2.dilate(image_1, ver_kernel, iterations=3)
    # cv2.imwrite("vertical.jpg", vertical_lines)
    # Plot the generated image
    plotting = plt.imshow(image_1, cmap='gray')
    # plt.show()

    image_2 = cv2.erode(img_bin, hor_kernel, iterations=3)
    horizontal_lines = cv2.dilate(image_2, hor_kernel, iterations=3)
    # cv2.imwrite("horizontal.jpg", horizontal_lines)
    # Plot the generated image
    plotting = plt.imshow(image_2, cmap='gray')
    # plt.show()

    # Combine horizontal and vertical lines in a new third image, with both having same weight.
    img_vh = cv2.addWeighted(vertical_lines, 0.5, horizontal_lines, 0.5, 0.0)
    # Eroding and thesholding the image
    img_vh = cv2.erode(~img_vh, kernel, iterations=2)
    thresh, img_vh = cv2.threshold(img_vh, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # cv2.imwrite("img_vh.jpg", img_vh)
    bitxor = cv2.bitwise_xor(img, img_vh)
    bitnot = cv2.bitwise_not(bitxor)
    # Plotting the generated image
    plotting = plt.imshow(bitnot, cmap='gray')
    # plt.show()

    # Detect contours for following box detection
    contours, hierarchy = cv2.findContours(img_vh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


    # cv2.drawContours(img, contours, -1, (0,255,0), 1)
    # plotting = plt.imshow(img, cmap='gray')
    # plt.show()

    def sort_contours(cnts, method="left-to-right"):
        reverse = False
        i = 0
        # handle if we need to sort in reverse
        if method == "right-to-left" or method == "bottom-to-top":
            reverse = True
        # handle if we are sorting against the y-coordinate rather than
        # the x-coordinate of the bounding box
        if method == "top-to-bottom" or method == "bottom-to-top":
            i = 1
        # construct the list of bounding boxes and sort them from top to
        # bottom
        boundingBoxes = [cv2.boundingRect(c) for c in cnts]
        cnts, boundingBoxes = zip(*sorted(zip(cnts, boundingBoxes),
                                          key=lambda b: b[1][i], reverse=reverse))
        # return the list of sorted contours and bounding boxes
        return cnts, boundingBoxes


    # Sort all the contours by top to bottom.
    # sourcery no-metrics
    contours, boundingBoxes = sort_contours(contours, method='top-to-bottom')

    # Creating a list of heights for all detected boxes
    heights = [boundingBoxes[i][3] for i in range(len(boundingBoxes))]
    # Get mean of heights
    mean = np.mean(heights)

    max_sum_w_and_h = dict()
    for i, max_countours in enumerate(contours):
        x, y, w, h = cv2.boundingRect(max_countours)
        max_sum_w_and_h[i] = w + h

    max_sum_w_and_h = {k: v for k, v in sorted(max_sum_w_and_h.items(), key=lambda item: item[1], reverse=True)}
    max_sum_w_and_h = dict(itertools.islice(max_sum_w_and_h.items(), 5))
    max_sum_w_and_h = {k: v for k, v in sorted(max_sum_w_and_h.items(), key=lambda item: item[0])}
    max_sum_w_and_h = dict(itertools.islice(max_sum_w_and_h.items(), 4))
    max_sum_w_and_h = {k: v for k, v in sorted(max_sum_w_and_h.items(), key=lambda item: item[1], reverse=True)}
    max_sum_w_and_h = dict(itertools.islice(max_sum_w_and_h.items(), count_of_tables + 1))
    max_sum_w_and_h = list(max_sum_w_and_h.values())

    # max_sum_w_and_h = sorted(max_sum_w_and_h, reverse=True)[:3]
    # Create list box to store all boxes in
    box = []
    # Get position (x,y), width and height for every contour and show the contour on image
    for i, c in enumerate(contours):
        x, y, w, h = cv2.boundingRect(c)
        if w + h in max_sum_w_and_h or h < min_height_of_cell or w < min_width_of_cell or i == del_cell:
            continue
        image = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        box.append([x, y, w, h])

    # cv2.imwrite("img_vh2.jpg", image)
    plotting = plt.imshow(image, cmap='gray')
    # plt.show()

    # Creating two lists to define row and column in which cell is located
    row = []
    column = []
    j = 0
    # Sorting the boxes to their respective row and column
    for i in range(len(box)):
        if i == 0:
            column.append(box[i])
            previous = box[i]
        else:
            if box[i][1] <= previous[1] + mean / 14:
                column.append(box[i])
                previous = box[i]
                if i == len(box) - 1:
                    row.append(column)
            else:
                row.append(column)
                column = []
                previous = box[i]
                column.append(box[i])
                if i == len(box) - 1:
                    row.append(column)

    dict_with_row_col = dict()
    for row_2, rows in enumerate(row, 1):
        for col, y_col in enumerate(rows, 1):
            dict_with_row_col[row_2, col] = y_col

    list_len_row = list()
    dict_len_row = dict()
    for count_row in range(len(row)):
        list_len_row.append(len(row[count_row]))
        dict_len_row[len(row[count_row])] = row[count_row]

    len_long_row = sorted(list_len_row)[-1]
    try:
        len_long_row_pred_max = sorted(set(list_len_row))[-2] if sorted(set(list_len_row))[-2] <= 7 else \
            sorted(set(list_len_row))[-3]
    except:
        len_long_row_pred_max = sorted(set(list_len_row))[-1]

    if len_long_row in dict_len_row:
        max_len_row = dict_len_row[len_long_row]
    if len_long_row_pred_max in dict_len_row:
        pred_max_len_row = dict_len_row[len_long_row_pred_max]

    # if count_row % 2 == 0:
    #     count_row = count_row - 1
    # print(row[count_row])
    center_one_table = [int(pred_max_len_row[j][0] + pred_max_len_row[j][2] / 2) for j in range(len(pred_max_len_row))
                        if row[0]]
    center_one_table = np.array(center_one_table)
    center_one_table.sort()

    center = [int(max_len_row[j][0] + max_len_row[j][2] / 2) for j in range(len(max_len_row)) if row[0]]
    center = np.array(center)
    center.sort()

    # Regarding the distance to the columns center, the boxes are arranged in respective order
    finalboxes = []
    for i in range(len(row)):
        lis = []
        for k in range(len_long_row):
            lis.append([])
        for j in range(len(row[i])):
            if len(row[i]) == len_long_row_pred_max or len(row[i]) == len_long_row_pred_max - 1:
                diff = abs(center_one_table - (row[i][j][0] + row[i][j][2] / 4))
            else:
                diff = abs(center - (row[i][j][0] + row[i][j][2] / 4))
            minimum = min(diff)
            indexing = list(diff).index(minimum)
            lis[indexing].append(row[i][j])
        finalboxes.append(lis)


    def find_score_each_word(file_name_cell, not_count_container_name=False):
        root = ET.ElementTree(ET.fromstring(file_name_cell))
        list_score = list()
        str_words_in_cell = ''
        for score in root.iter():
            keys = score.attrib.keys()
            if 'title' in list(keys) and re.findall("x_wconf", score.attrib['title']):
                # print(len(score.text), score.text)
                find_container_name = re.findall('Ko[a-z]', score.text)
                if not_count_container_name:
                    if not find_container_name and not score.text == ' ' and len(score.text) > 1:
                        list_score.append(int(score.attrib['title'].split()[-1]))
                    str_words_in_cell += score.text + ' '
                else:
                    if not score.text == ' ':
                        list_score.append(int(score.attrib['title'].split()[-1]))
                    str_words_in_cell += score.text + ' '
        return str_words_in_cell, list_score


    def find_row_col(row, x, y):
        for row_new, rows in enumerate(row, 1):
            rows.sort(key=lambda x: x[0])
            for let in range(len(rows)):
                if x == rows[let][1]:
                    for col_new, col in enumerate(rows, 1):
                        if y == col[0]:
                            return row_new, col_new


    # row_new = 0
    # col_new = 0
    file_name_cell = 'name'
    list_col = ['']
    col_number = False
    col_number_goods = False
    activation_for_short_word = False
    row_col = tuple()
    for i in range(len(finalboxes)):
        for j in range(len(finalboxes[i])):
            inner = ''
            if len(finalboxes[i][j]) == 0:
                outer.append(' ')
            else:
                table_dict = dict()
                ocr_json['type'] = 'table'
                for k in range(len(finalboxes[i][j])):
                    y, x, w, h = finalboxes[i][j][k][0], finalboxes[i][j][k][1], finalboxes[i][j][k][2], \
                                 finalboxes[i][j][k][3]
                    finalimg = bitnot[x + indent_x_text_of_cells:x + h - indent_x_text_of_cells,
                               y + indent_y_text_of_cells:y + w - indent_y_text_of_cells]
                    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))
                    # border = cv2.copyMakeBorder(finalimg, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=[255, 255])
                    resizing = cv2.resize(finalimg, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
                    dilation = cv2.dilate(resizing, kernel, iterations=1)
                    erosion = cv2.erode(dilation, kernel, iterations=1)
                    erosion = cv2.fastNlMeansDenoising(erosion, None, 20, 7, 21)
                    if 1450 > w > 1100:
                        text = pytesseract.image_to_pdf_or_hocr(erosion, extension='hocr',
                                                                lang="eng")
                        list_score_and_out = find_score_each_word(text, not_count_container_name=True)
                        list_score = list_score_and_out[1]
                        out = list_score_and_out[0].replace('.', ',')
                    else:
                        if activation_for_short_word or list_col[-1] == col_number:
                            text = pytesseract.image_to_pdf_or_hocr(erosion, extension='hocr',
                                                                    config='--oem 3 --psm 12',
                                                                    lang="rus+eng")
                            activation_for_short_word = False
                        elif list_col[-1] == col_number_goods:
                            text = pytesseract.image_to_pdf_or_hocr(erosion, extension='hocr', config='--oem 3 --psm 6',
                                                                    lang="rus+eng")
                        else:
                            text = pytesseract.image_to_pdf_or_hocr(erosion, extension='hocr',
                                                                    config=config_for_pytesseract,
                                                                    lang="rus+eng")
                        list_score_and_out = find_score_each_word(text)
                        list_score = list_score_and_out[1]
                        out = list_score_and_out[0]

                    inner = inner + " " + out.strip()
                    inner = inner.translate({ord(c): " " for c in "!@#$%^&*()[]{};<>?\|`~-=_+"})
                    inner = inner.replace('', ' ').replace('\n', ' ').replace('  ', ' ').replace('????????',
                                                                                                  '?????? ??').replace(
                        '??$ ????????????', 'AS ROSALIA').replace('B??????', '?????? ??').replace('Be????', '?????? ??')
                    print(inner)
                    if inner != '' and inner != ' ' and len(inner) > 2:
                        table_dict['text'] = inner.strip()
                        row_col = find_row_col(row, x, y)
                        table_dict['row'] = row_col[0]
                        table_dict['col'] = row_col[1]
                        table_dict['xmin'] = y
                        table_dict['ymin'] = x
                        table_dict['xmax'] = y + w
                        table_dict['ymax'] = x + h
                        table_dict['score'] = np.mean(list_score) if len(list_score) != 0 else None
                        table_dict['std'] = np.std(list_score) if len(list_score) != 0 else None
                        table_dict['validation'] = True if np.mean(list_score) > 85 else False
                        table_list.append(table_dict)
                    try:
                        list_col.append(row_col[1])
                    except:
                        pass
                    if re.findall('????????????????????????????', out):
                        activation_for_short_word = True
                    elif re.findall('??????', out) or re.findall('??????', out):
                        col_number = row_col[1] - 1
                    elif re.findall('??? ????????????', out) or re.match('????????????', out):
                        col_number_goods = row_col[1] - 1

                ocr_json['cells'] = table_list
                outer.append(inner)

    predection_box.append(ocr_json)

    # Creating a dataframe of the generated OCR list
    arr = np.array(outer)
    dataframe = pd.DataFrame(arr.reshape(len(row), len_long_row))
    # insert_line = pd.DataFrame(ship_voyage)
    # dataframe = pd.concat([insert_line, dataframe])
    print(dataframe.to_string())

    dataframe.to_csv(output_directory_csv + '/' + os.path.basename(file + '.csv'), encoding='utf-8', index=False)
    file_json_save = output_directory + '/' + os.path.basename(file + '.json')

    predicted_boxes_dict['file_name'] = os.path.basename(file)
    predicted_boxes_dict['predicted_box'] = predection_box
    with open(file_json_save, 'w', encoding='utf-8') as f:
        json.dump(predicted_boxes_dict, f, ensure_ascii=False, indent=4)

    print(os.path.basename(config_yaml_file))

    # try:
    #     conn = psycopg2.connect(database=database, user=user, password=password, host=host, port=port)
    #     cur = conn.cursor()
    #     data_json = json.dumps(predicted_boxes_dict, ensure_ascii=False, indent=4)
    #     sql = f"INSERT INTO {table} (data_json, image, url_image) VALUES (%s, %s, %s)"
    #     val = (data_json, f'upload/{os.path.basename(file_json_save)}', f'{os.path.basename(file_json_save)}')
    #     cur.execute(sql, val)
    #     cur.close()
    #     conn.commit()
    #     conn.close()
    # except Exception as exception:
    #     print(exception)
except:
    file_json_save = output_directory + '/' + os.path.basename(file + '.json')
    predicted_boxes_dict['file_name'] = os.path.basename(file)
    predicted_boxes_dict['predicted_box'] = predection_box
    if predection_box:
        with open(file_json_save, 'w', encoding='utf-8') as f:
            json.dump(predicted_boxes_dict, f, ensure_ascii=False, indent=4)

    # try:
    #     conn = psycopg2.connect(database=database, user=user, password=password, host=host, port=port)
    #     cur = conn.cursor()
    #     data_json = json.dumps(predicted_boxes_dict, ensure_ascii=False, indent=4)
    #     sql = f"INSERT INTO {table} (data_json, image, url_image) VALUES (%s, %s, %s)"
    #     val = (data_json, f'upload/{os.path.basename(file_json_save)}', f'{os.path.basename(file_json_save)}')
    #     cur.execute(sql, val)
    #     cur.close()
    #     conn.commit()
    #     conn.close()
    # except Exception as exception:
    #     print(exception)
