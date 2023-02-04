"""
Objective
==============
Generate the caption for an image using two NN (CNN and NLP).
https://data-flair.training/blogs/python-based-project-image-caption-generator-cnn/
"""


# Library import
# ===============
import torchvision.models
import torch.nn
import os
import re
import time



# Variable definition
# ===================

root_path_txt_data = "/Users/swmoeller/python/prj_image_captioning_e2e/data/05_raw/Flickr8k_text"
file_image_description = "Flickr8k.token.txt"
requirement_file = "/Users/swmoeller/python/prj_image_captioning_e2e/data/20_processed/requirement.txt"

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
    :return: dictionary containing for each key all the captions
    :rtype: dict
    """
    caption_dict = {}
    file_image_caption = os.path.join(in_data_path_text, in_filename)
    print("[INFO] Processing text input file: ", file_image_caption)
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


def clean_dict(in_dict: dict):
    """
    Cleaning of dictionaries using regex
    - lower casing, removing punctuations and words containing numbers
        -- regular expression HowTo: https://docs.python.org/3/howto/regex.html
    
    :param in_dict: dictionary to be cleaned
    :type in_dict: dict
    :return: Dictionary cleaned from punctuations, a & A and 's
    :rtype: dict
    """
    for keys, values in in_dict.items():
        
        # 1. read all values from first key into list
        list_value = []
        for items in range(len(values)):
            list_value.append(values[items])
        
        # 2. clean each entry of each element in the list
        for item in range(len(list_value)):
            re_pattern = re.compile(r"""
            \b[aA's]\b        # single a or A enclosed by blank
            |               # or
            [\.,;:!]        # any punctuation
            |               # or
            's              # 's attached to word
            |
            /gi             # all single a, A (enclosed by blanks) and punctuations & 's
            with global search ignoring case (lower, upper case)
            """, re.VERBOSE)
            list_value[item] = re_pattern.sub("", list_value[item]).lower()
            list_value[item] = re.sub("\s{2,}", " ", list_value[item]).strip() # delete trailing and leading spaces
            # as well as multiple spaces

        # 3. delete all values from first key
        # 4. Write back cleaned list to first key
        in_dict[keys] = list_value
        
    # 5. Go to next key
    print("[INFO] Clean dictionary generated; removal of punctuation and 'a, A'")
    return in_dict


def unique_vocabulary(in_clean_dict: dict):
    """
    This is a simple function that will separate all the unique words and create the vocabulary from all the descriptions.
    :param in_clean_dict: dictionary containing the images (keys) and their captions
    :type in_clean_dict: dict
    :return: list of unique words used in captions
    :rtype: list
    """
    list_of_words = []
    # 1. Call first key
    for keys, values in in_clean_dict.items():
        
        # 2. Read out the list containing all captions
        for captions in values:
            
            # 3. Go through each item in list (string) and split it
            for word in captions.split():
    
                # 4. If a word in the string does not already exist in vocab, append it
                if word not in list_of_words:
                    list_of_words.append(word)
    print("[INFO] List of unique words (out of captions) generated.")
    return list_of_words


def save_descriptions(in_clean_dict: dict):
    """
    This function will create a list of all the descriptions that have been preprocessed and store them into a file.
    
    :param in_clean_dict:
    :type in_clean_dict:
    :return:
    :rtype:
    """
    outfile = open(requirement_file,"w")
    # output the header row
    outfile.write("image\tcaption\n")
    # output image name and each caption line-by-line
    for keys, values in clean_dictionary.items():
        for items in range(len(values)):
            row_string = "{}\t{}\n".format(keys, values[items])
            outfile.write(row_string)
    
    outfile.close()
    print("[INFO] requirement.txt generated.")

    return


# MODEL definition
# ================
frcnn_model = torchvision.models.get_model("resnet50", weights="DEFAULT")

#  replace fc with Identity, which just returns the input as the output, and since the features are its input,
#  the output of the entire model will be the features.
frcnn_model.fc = torch.nn.Identity()

# ===================
# MAIN
# ===================
print("\n[START] Start of execution {}\n".format(time.strftime("%H:%M:%S")))
description_dictionary = generate_dict(root_path_txt_data, file_image_description)
clean_dictionary = clean_dict(description_dictionary)
unique_words = unique_vocabulary(clean_dictionary)
save_descriptions((clean_dictionary))


# print("[INFO] Parameter of selected model for image recognition", frcnn_model)
print("\n[END] End of execution: ", time.strftime("%H:%M:%S"))
