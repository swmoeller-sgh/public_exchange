import json
import pandas as pd

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
    key_amount = len(label_dict[single_model]["0.9"].keys())-3
    if key_amount > max_keys[1]:
        max_keys = [single_model, key_amount]
print(max_keys)
print(label_dict[max_keys[0]]["0.9"])

total_sum = 0

# generate a filtered dict. according to model and confidence level as well as excluding descriptive information such
# as model, etc.
filtered_dict = dict(filter(lambda item: "[" not in item[0], label_dict[max_keys[0]]["0.9"].items()))
print(filtered_dict)

# total no of objects
for value in filtered_dict.values():
    total_sum = total_sum + value
print(total_sum)

# add % of objects on total to dataframe
keyword_matrix_pd = pd.DataFrame.from_dict(filtered_dict, orient='index')
keyword_matrix_pd.sort_index(axis=1, inplace=True)
keyword_matrix_pd.columns = ["object_amount"]
keyword_matrix_pd["percentage"] = keyword_matrix_pd["object_amount"] / total_sum
keyword_matrix_pd["weighted occasion"] = keyword_matrix_pd["percentage"] * max_keys[1]
keyword_matrix_pd["weighted occasion"] = keyword_matrix_pd["weighted occasion"].round(decimals = 0)
keyword_matrix_pd["weighted occasion"] = keyword_matrix_pd["weighted occasion"].astype("int")
keyword_matrix_pd["label"] = keyword_matrix_pd.index

# read the labels and generate a list out of it
# (count the total counts and put label-specific counts into relation)
# repeat a single keyword according to the weighting in the docstring
keyword_matrix_pd["keyword_object_amount"] = (keyword_matrix_pd["label"] + " ")* keyword_matrix_pd["weighted occasion"]

print(keyword_matrix_pd)

# generate a string with keywords
keyword_list = keyword_matrix_pd["keyword_object_amount"].values.tolist()
keyword_list_not_weighted = keyword_matrix_pd["label"]

print(keyword_list)

# Generate the caption
# --------------------

# read the keyword list
# NLP generates the caption

from keytotext import pipeline

# Load the base pre-trained T5 model
# It will download three files: 1. config.json, 2. tokenizer.json, 3. pytorch_model.bin (~850 MB)
nlp = pipeline("k2t-base")

# Configure the model parameters
config = {"do_sample": True, "num_beams": 4, "no_repeat_ngram_size": 3, "early_stopping": True}

# Provide list of keywords into the model as input
print(nlp(keyword_list, **config))
print(nlp(keyword_list_not_weighted, **config))