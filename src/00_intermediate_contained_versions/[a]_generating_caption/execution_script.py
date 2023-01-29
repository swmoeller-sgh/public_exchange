"""
Purpose
=======
Detection and classification of objects within an image.


Use the labels to train o Huggingface NLP-model generating a docstring based on selected keywords using a seq2seq
model (T5/BART).

\\

Reference
=========
- Object detection and classification: "https://pytorch.org/vision/stable/models.html"
- Seq2Seq model (base tutorial): https://github.com/NielsRogge/Transformers-Tutorials/blob/master/T5/Fine_tune_CodeT5_for_generating_docstrings_from_Ruby_code.ipynb
- Execute model on AWS: https://github.com/huggingface/notebooks/blob/main/sagemaker/01_getting_started_pytorch/sagemaker-notebook.ipynb

"""


# Library import
# ===============

import ANN_models as ann_models

from torchvision.io.image import read_image
from torchvision.utils import draw_bounding_boxes

import torch

import pandas as pd

import time
from datetime import datetime

import json

from keytotext import pipeline          # NLP model to convert list of keywords into docstring

from torchvision.transforms.functional import to_pil_image


# Variable definition
# ===================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
option_dict_cnn = {}

MIN_CONFIDENCE_LEVEL_RANGE = [0.7, 0.8, 0.9, 0.95, 0.99]
img = read_image(
    "/Users/swmoeller/python/X_DataDir/datasets/LR_samples_for_testing/20170903_003_IT_Rome_Sightseeing in Rome.jpg")
# img = read_image("/Users/swmoeller/Downloads/000000547383.jpg")

# generate list of options from keys in cnn model dictionary
for k in ann_models.MODEL_CATALOG_CNN.keys():
    option_dict_cnn[k] = k


# CLASS definition
# ================
class DateTimeEncoder(json.JSONEncoder):
    """
    Convert date/time into string, as date/time is not accepted in serializing
    """
    def default(self, o):
        if isinstance(o, datetime):
            return o.isoformat()

        return super().default(o)

# FUNCTION definition
# ===================
def selectFromDict(options, name):
    """
    Generates a menu (based on options given), which allows the user to select one of the options.
    :param options: dictionary containing the different options
    :type options: dict
    :param name: description of choice
    :type name: str
    :return: Selected element
    :rtype: str
    """
    index = 0
    indexValidList = []
    print('Select a ' + name + ':')
    for optionName in options:
        index = index + 1
        indexValidList.extend([options[optionName]])
        print(str(index) + ') ' + optionName)
    inputValid = False
    while not inputValid:
        inputRaw = input(name + ': ')
        inputNo = int(inputRaw) - 1
        if inputNo > -1 and inputNo < len(indexValidList):
            selected = indexValidList[inputNo]
            print('\n[INFO] Selected ' + name + ': ' + selected)
            inputValid = True
            break
        else:
            print('\n[WARNING] Please select a valid ' + name + ' number')
    
    return selected

def image_processing(IN_model_choice, IN_startTime):
    
    startTime = IN_startTime
    
    # Step 1: Initialize model with the best available weights
    weights = ann_models.MODEL_CATALOG_CNN[IN_model_choice]["weight"].DEFAULT
    print("[INFO] Weights selected", weights)
    
    model = ann_models.MODEL_CATALOG_CNN[IN_model_choice]["model"](weights=weights,
                                                                box_score_thresh=MIN_CONFIDENCE_LEVEL)
    # box_score_thresh: during inference, only return proposals with a classification score greater than box_score_thresh
    model.eval()
    
    # Step 2: Initialize the inference transforms
    preprocess = weights.transforms()
    
    # Step 3: Apply inference preprocessing transforms
    batch = [preprocess(img)]
    
    # Step 4: Use the model and visualize the prediction
    prediction = model(batch)[0]
    labels = [weights.meta["categories"][i] for i in prediction["labels"]]
    box = draw_bounding_boxes(img, boxes=prediction["boxes"],
                              font="Arial",
                              labels=labels,
                              colors="red",
                              width=4, font_size=20)
    # im = to_pil_image(box.detach())
    # im.show()
    print("[INFO] Identified labels:\n", labels)
    
    # if model not yet registered in dictionary, it is being registered
    if IN_model_choice not in label_dict:
        label_dict[IN_model_choice] = {}
        print(label_dict)
    
    # if the minimum confidence level was not yet tested, a new entry is being established
    if MIN_CONFIDENCE_LEVEL not in label_dict[IN_model_choice]:
        label_dict[IN_model_choice][str(MIN_CONFIDENCE_LEVEL)] = {}
    
    for label in labels:
        print("[INFO] Recognized element (label):", label)
        if label in label_dict[IN_model_choice][str(MIN_CONFIDENCE_LEVEL)]:
            print(f"[INFO] increase count of label '{label}' by one")
            label_dict[IN_model_choice][str(MIN_CONFIDENCE_LEVEL)][label] = label_dict[IN_model_choice][
                                                                            str(MIN_CONFIDENCE_LEVEL)][label] + 1
            print("[INFO] Updated dictionary: ", label_dict)
        else:
            print("[INFO] New entry for label ", label)
            label_dict[IN_model_choice][str(MIN_CONFIDENCE_LEVEL)][label] = 1
            print("[INFO] Updated dictionary: ", label_dict)
    label_dict[IN_model_choice][str(MIN_CONFIDENCE_LEVEL)]["[0] time run"]= startTime
    label_dict[IN_model_choice][str(MIN_CONFIDENCE_LEVEL)]["[1] weight"]= weights
    label_dict[IN_model_choice][str(MIN_CONFIDENCE_LEVEL)]["[2] confidence level"]= MIN_CONFIDENCE_LEVEL
    
    return label_dict

# ===================
# MAIN
# ===================

model_choice = selectFromDict(option_dict_cnn, 'CNN Model')

for MIN_CONFIDENCE_LEVEL in MIN_CONFIDENCE_LEVEL_RANGE:
    startTime = datetime.fromtimestamp(time.time())
    
    # Open previous dictionaries (i.e. detection and classification record)
    try:
        with open('label_result_dict.json') as infile:
            label_dict = json.load(infile)
            print("[INFO] Existing dictionary loaded:\n", label_dict)
    except:
        label_dict = {}

    # Generate a list of models to be evaluated (multiple or just one)
    if model_choice == "Test all models":
        model_choice_list = list(ann_models.MODEL_CATALOG_CNN.keys())
    else:
        model_choice_list = [model_choice]
    
    print("[INFO] List of models ", model_choice_list)

    # When selected "Test all models" from options, no model obviously could be found in the catalog. Therefore the
    # selection is being excluded by checking, if a model name exists
    
    for single_model in model_choice_list:
        if "model" in ann_models.MODEL_CATALOG_CNN[single_model]:
            print("\n\n[INFO] Processed model ", single_model)
            image_processing(single_model, startTime)
    
    print(label_dict)
    
    # generate a panda dataframe
    model_evaluation_pd = pd.DataFrame.from_dict(label_dict, orient='index')
    model_evaluation_pd.sort_index(axis=1, inplace= True)
    
    print(model_evaluation_pd)
    model_evaluation_pd.to_excel("model_evaluation.xlsx")
 
    # Save the dictionary as json
    with open('label_result_dict.json', 'w') as fp:
        json.dump(label_dict, fp, sort_keys=True, indent=4, cls=DateTimeEncoder, default = str)


# Generate the docstring input
# ----------------------------

# Open dictionary
try:
    with open('label_result_dict.json') as infile:
        label_dict = json.load(infile)
        print("[INFO] Existing dictionary loaded:\n", label_dict)
except:
    print("[ERROR] opening dictionary")

# run through the dictionary and find the model with the maximum number of labels check the number of labels and
# detected objects (not starting with [x]) within the confidence level = 0.9 by model
max_keys = ["none", 0]

for single_model in label_dict:
    key_amount = len(label_dict[single_model]["0.9"].keys()) - 3
    if key_amount > max_keys[1]:
        max_keys = [single_model, key_amount]
print(max_keys)
print(label_dict[max_keys[0]]["0.9"])

# generate a filtered dict. according to model and confidence level as well as excluding descriptive information such
# as model, etc.
filtered_dict = dict(filter(lambda item: "[" not in item[0], label_dict[max_keys[0]]["0.9"].items()))
print(filtered_dict)

# total no of objects
total_sum = 0
for value in filtered_dict.values():
    total_sum = total_sum + value
print(total_sum)

# add % of objects on total to dataframe
keyword_matrix_pd = pd.DataFrame.from_dict(filtered_dict, orient='index')
keyword_matrix_pd.sort_index(axis=1, inplace=True)
keyword_matrix_pd.columns = ["object_amount"]
keyword_matrix_pd["percentage"] = keyword_matrix_pd["object_amount"] / total_sum

# calculate and convert column into integer value
keyword_matrix_pd["weighted occasion"] = keyword_matrix_pd["percentage"] * max_keys[1]
keyword_matrix_pd["weighted occasion"] = keyword_matrix_pd["weighted occasion"].round(decimals=0)
keyword_matrix_pd["weighted occasion"] = keyword_matrix_pd["weighted occasion"].astype("int")
keyword_matrix_pd["label"] = keyword_matrix_pd.index

# read the labels and generate a list out of it
# (count the total counts and put label-specific counts into relation)
# repeat a single keyword according to the weighting in the docstring

# filter all labels excluding the artificial labels such as model name, etc.
keyword_matrix_pd["keyword_object_amount"] = (keyword_matrix_pd["label"] + " ") * keyword_matrix_pd[
    "weighted occasion"]

print(keyword_matrix_pd)

# generate a string with keywords
keyword_list = keyword_matrix_pd["keyword_object_amount"].values.tolist()
keyword_list_not_weighted = keyword_matrix_pd["label"]

print(keyword_list)

# Generate the caption
# --------------------

# read the keyword list and NLP generates the caption

# Load the base pre-trained T5 model
# It will download three files: 1. config.json, 2. tokenizer.json, 3. pytorch_model.bin (~850 MB)
nlp = pipeline("k2t-base")

# Configure the model parameters
config = {"do_sample": True, "num_beams": 4, "no_repeat_ngram_size": 3, "early_stopping": True}

# Provide list of keywords into the model as input
print(nlp(keyword_list, **config))
print(nlp(keyword_list_not_weighted, **config))
