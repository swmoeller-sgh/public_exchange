"""
Objective
==============
Generate the caption for an image using two NN (CNN and NLP).
"""


# Library import
# ===============
import torchvision.models
import torch.nn
import os


# Variable definition
# ===================

root_path_txt_data = "/Users/swmoeller/python/prj_image_captioning_e2e/data/05_raw/Flickr8k_text"
file_image_description = "Flickr8k.token.txt"

search_string = "#"         # string to be found in text file to identify captions

# CLASS definition
# ================


# FUNCTION definition
# ===================
def generate_dict(in_data_path_text: str, in_filename: str):
    """
    Generate description description_dictionary for look up of possible captions (if more than one)
        
        read line
        find #
        treat everything before # as key
        treat everything after the position of # +2 as caption
        check if key exists and
        if exists, add caption
        if not exists, add key + caption
        clean sentences from \n and \t
        
    {"image_name1":["caption1", "caption2", "caption3"], "image_name2":...}
    :param in_data_path_text:
    :type in_data_path_text:
    :param in_filename: name of text file containing image name and captions
    :type in_filename: str
    """
    caption_dict = {}
    file_image_caption = os.path.join(in_data_path_text, in_filename)
    print(file_image_caption)
    if os.path.isfile(path=file_image_caption):
        dic_raw = open(file=file_image_caption, mode="r")

        for aline in dic_raw.readlines():
            # check if string present on a current line
            if aline.find(search_string) != -1:
                pos_string = aline.find(search_string)
                
                key = aline[0:pos_string]
                value = aline[pos_string+3:].rstrip("\n\t")
                if key in caption_dict:
                    caption_dict[key].append(value)
                else:
                    caption_dict[key] = [value]
        dic_raw.close()
    else:
        print(f"{in_data_path_text} or {in_filename} do not exist!")
        exit()

    print("[INFO] Dictionary with image names and their captions generated.")
    return caption_dict


# MODEL definition
# ================
frcnn_model = torchvision.models.get_model("resnet50", weights="DEFAULT")

#  replace fc with Identity, which just returns the input as the output, and since the features are its input,
#  the output of the entire model will be the features.
frcnn_model.fc = torch.nn.Identity()

# ===================
# MAIN
# ===================
description_dictionary = generate_dict(root_path_txt_data, file_image_description)


# **cleaning\_text( descriptions) –** This function takes all descriptions and performs data cleaning.

#  **text\_vocabulary( descriptions ) –** This is a simple function that will separate all the unique words
# **save\_descriptions( descriptions, filename )

# print("[INFO] Parameter of selected model for image recognition", frcnn_model)
