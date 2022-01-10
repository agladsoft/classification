import glob
import os
import shutil
import torch
from PIL import Image
from transformers import LayoutLMv2FeatureExtractor, LayoutLMv2Tokenizer, LayoutLMv2Processor


dir_classific = 'dir_classific'
dir_line = 'line'
dir_port = 'port'
dir_port_other = 'two_page_port'
dir_garbage = 'garbage'

if not os.path.exists(dir_classific):
    os.makedirs(os.path.join(dir_classific, dir_line))
    os.makedirs(os.path.join(dir_classific, dir_port))
    os.makedirs(os.path.join(dir_classific, dir_port_other))
    os.makedirs(os.path.join(dir_classific, dir_garbage))


def classific_imgs():
    PATH = "model_for_classification_documents.pt"
    model = torch.load(PATH)
    labels = ['line', 'port', 'two_page_port', 'garbage']
    id2label = {v: k for v, k in enumerate(labels)}
    print(id2label)
    feature_extractor = LayoutLMv2FeatureExtractor()
    tokenizer = LayoutLMv2Tokenizer.from_pretrained("microsoft/layoutlmv2-base-uncased")
    processor = LayoutLMv2Processor(feature_extractor, tokenizer)

    for file_name in sorted(glob.glob(f"dir_img/*.jpg")):
        foo = Image.open(file_name)
        (width, height) = foo.size
        foo = foo.resize((width // 4, height // 4), Image.ANTIALIAS)
        foo.save("crop.jpg", optimize=True, quality=95)

        image = Image.open("crop.jpg")
        image = image.convert("RGB")

        encoded_inputs = processor(image, return_tensors="pt")
        outputs = model(**encoded_inputs)
        print(os.path.basename(file_name))
        print(outputs)

        logits = outputs.logits
        predicted_class_idx = logits.argmax(-1).item()
        predict = id2label[predicted_class_idx]
        print("Predicted class:", predict)

        if predict == 'line':
            shutil.move(file_name, f"{dir_classific}/{dir_line}")
        elif predict == 'port':
            shutil.move(file_name, f"{dir_classific}/{dir_port}")
        elif predict == 'two_page_port':
            shutil.move(file_name, f"{dir_classific}/{dir_port_other}")
        elif predict == 'garbage':
            shutil.move(file_name, f"{dir_classific}/{dir_garbage}")

    os.remove("crop.jpg")


classific_imgs()
