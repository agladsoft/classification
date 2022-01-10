from PIL import Image
import pytesseract
import numpy as np
from transformers import LayoutLMv2FeatureExtractor, LayoutLMv2Tokenizer, LayoutLMv2Processor
import pandas as pd
import os
from datasets import Dataset
from datasets import Features, Sequence, ClassLabel, Value, Array2D, Array3D
import torch
from transformers import LayoutLMv2ForSequenceClassification
from transformers import AdamW
from tqdm.notebook import tqdm


# image_original = Image.open("categories_doc/black_si_trance_line/7534  MAERSK GIRONDE В 04.11.2021.pdf-038.jpg")
# image_original = image_original.resize((550, 600), Image.ANTIALIAS)
# image_original.save("7534  MAERSK GIRONDE В 04.11.2021.pdf-038.jpg", optimize=True, quality=95)
image = Image.open("7534  MAERSK GIRONDE В 04.11.2021.pdf-038.jpg")
image = image.convert("RGB")
print(image)

NCCL_ASYNC_ERROR_HANDLING=1

ocr_df = pytesseract.image_to_data(image, output_type='data.frame')
ocr_df = ocr_df.dropna().reset_index(drop=True)
float_cols = ocr_df.select_dtypes('float').columns
ocr_df[float_cols] = ocr_df[float_cols].round(0).astype(int)
ocr_df = ocr_df.replace(r'^\s*$', np.nan, regex=True)
words = ' '.join([word for word in ocr_df.text if str(word) != 'nan'])
print(words)


feature_extractor = LayoutLMv2FeatureExtractor()
tokenizer = LayoutLMv2Tokenizer.from_pretrained("microsoft/layoutlmv2-base-uncased")
processor = LayoutLMv2Processor(feature_extractor, tokenizer)
encoded_inputs = processor(image, return_tensors="pt")


for k, v in encoded_inputs.items():
    print(k, v.shape)
processor.tokenizer.decode(encoded_inputs.input_ids.squeeze().tolist())


dataset_path = "categories_doc"
labels = [label for label in os.listdir(dataset_path)]
id2label = {v: k for v, k in enumerate(labels)}
label2id = {k: v for v, k in enumerate(labels)}
print(label2id)
print(id2label)


images = []
labels = []
for label_folder, _, file_names in os.walk(dataset_path):
    if label_folder != dataset_path:
        label = label_folder[15:]
        for _, _, image_names in os.walk(label_folder):
            relative_image_names = []
            for image_file in image_names:
                relative_image_names.append(dataset_path + "/" + label + "/" + image_file)
                # break
            images.extend(relative_image_names)
            labels.extend([label] * len(relative_image_names))

data = pd.DataFrame.from_dict({'image_path': images, 'label': labels})
print(data.head())
data.to_csv("classification_data.csv")

labels = set(labels)
labels = list(labels)
# read dataframe as HuggingFace Datasets object
dataset = Dataset.from_pandas(data)
print(dataset)

# we need to define custom features
features = Features({
    'image': Array3D(dtype="int64", shape=(3, 224, 224)),
    'input_ids': Sequence(feature=Value(dtype='int64')),
    'attention_mask': Sequence(Value(dtype='int64')),
    'token_type_ids': Sequence(Value(dtype='int64')),
    'bbox': Array2D(dtype="int64", shape=(512, 4)),
    'labels': ClassLabel(num_classes=len(labels), names=labels),
})


def preprocess_data(examples):
    # take a batch of images
    images = [Image.open(path).convert("RGB") for path in examples['image_path']]
    encoded_inputs = processor(images, padding="max_length", truncation=True)
    # add labels
    encoded_inputs["labels"] = [label2id[label] for label in examples["label"]]
    return encoded_inputs


encoded_dataset = dataset.map(preprocess_data, remove_columns=dataset.column_names, features=features,
                              batched=True, batch_size=1)
encoded_dataset.set_format(type="torch")

dataloader = torch.utils.data.DataLoader(encoded_dataset, batch_size=1)


batch = next(iter(dataloader))

for k, v in batch.items():
    print(k, v.shape)
processor.tokenizer.decode(batch['input_ids'][0].tolist())
print(id2label[batch['labels'][0].item()])


device = torch.device("cpu")
model = LayoutLMv2ForSequenceClassification.from_pretrained("microsoft/layoutlmv2-base-uncased",
                                                            num_labels=len(labels))
model.to(device)


optimizer = AdamW(model.parameters(), lr=5e-5)
global_step = 0
num_train_epochs = 10
t_total = len(dataloader) * num_train_epochs  # total number of training steps

# put the model in training mode
model.train()
for epoch in range(num_train_epochs):
    print("Epoch:", epoch)
    running_loss = 0.0
    correct = 0
    for batch in tqdm(dataloader):
        # forward pass
        outputs = model(**batch)
        loss = outputs.loss

        running_loss += loss.item()
        predictions = outputs.logits.argmax(-1)
        correct += (predictions == batch['labels']).float().sum()

        # backward pass to get the gradients
        loss.backward()

        # update
        optimizer.step()
        optimizer.zero_grad()
        global_step += 1

    print("Loss:", running_loss / batch["input_ids"].shape[0])
    accuracy = 100 * correct / len(data)
    print("Training accuracy:", accuracy.item())


# prepare image for the model
encoded_inputs = processor(image, return_tensors="pt")
# make sure all keys of encoded_inputs are on the same device as the model
for k, v in encoded_inputs.items():
    encoded_inputs[k] = v.to(model.device)

# save model
PATH = "model_for_classification_documents_3.pt"
torch.save(model, PATH)

# forward pass
outputs = model(**encoded_inputs)
print(outputs)

logits = outputs.logits
print(logits.shape)

predicted_class_idx = logits.argmax(-1).item()
print(predicted_class_idx)
print("Predicted class:", id2label[predicted_class_idx])