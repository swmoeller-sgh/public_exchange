"""
Objective
==============
Generate the caption for an image using two NN (CNN and NLP).
https://data-flair.training/blogs/python-based-project-image-caption-generator-cnn/

1. Load the image and extract its features: Use a pre-trained deep learning model such as VGG or ResNet to extract the
features from the image and store the features in a variable.

2. Preprocess the captions: Tokenize the captions and convert them into numerical representations using a word
embedding.

3. Prepare the data: Split the data into training and testing sets, and create data loaders to feed the data into the
model during training and evaluation.

4. Define the LSTM model: Define the LSTM model architecture with PyTorch. The input to the LSTM should be the image
features and the output should be the predicted caption.

5. Train the model: Train the LSTM model using the training data and the PyTorch built-in loss functions and optimizers.

6. Evaluate the model: Evaluate the performance of the model on the testing data. You can use metrics such as BLEU or
ROUGE to evaluate the quality of the generated captions.

7. Generate captions: Use the trained LSTM model to generate captions for new images.



"""


# Library import
# ===============
import torchvision.models
from torchvision import transforms
import transformers

import torch.nn

import os
import re
import time
import cv2

import pickle

import numpy as np

import torch

# Variable definition
# ===================

# INPUT data
# ----------
path_2_text_data = "/Users/swmoeller/python/prj_image_captioning_e2e/data/05_raw/Flickr8k_text"
path_image_data = "/Users/swmoeller/python/prj_image_captioning_e2e/data/05_raw/Flicker8k_Dataset"
file_image_flickr_description = "Flickr8k.token.txt"
file_train_img_lst = "Flickr_8k.trainImages.txt"


# TRAINING directory
# ------------------
path_train_dir = "/Users/swmoeller/python/prj_image_captioning_e2e/data/05_raw/Flicker8k_Dataset"


# OUTPUT data
# -----------
path_2_processed_data = "/Users/swmoeller/python/prj_image_captioning_e2e/data/20_processed/"
file_image_clean_caption = "description.txt"
file_image_features = "feature.p"
file_vocabulary = "vocabulary.txt"
path_tokenizer = 'modified_tokenizer'


# IMAGE related
# --------------

search_string = "#"         # string to be found in text file to identify captions


# TEXT related
# ------------

special_tokens = ["<CLS>", "<SEP>"]         # special tokens to be added to the Tokenizer (in this case HuggingFaces)


# CLASS definition
# ================

class ResNet50Features(torch.nn.Module):
    def __init__(self):
        super(ResNet50Features, self).__init__()
        resnet50_weights = torchvision.models.ResNet50_Weights.DEFAULT
        self.resnet50 = torchvision.models.resnet50(weights=resnet50_weights)
        # Freeze all layers to prevent backpropagation
        for param in self.resnet50.parameters():
            param.requiresGrad = False
        # Replace the fully connected layer with an identity function
        self.resnet50.fc = torch.nn.Identity()

    def forward(self, x):
        self.eval()
        with torch.no_grad():
            x = self.resnet50(x)
        return x


class CNN_and_NLP(torch.nn.Module):
    """
    The number of hidden layers
    The number of units in a hidden layer
    Activation functions performed at the various layers
    The loss function that we try to optimize for
    The learning rate associated with the neural network
    The batch size of data leveraged to build the neural network
    The number of epochs of forward and back-propagation

CNN, NLP model
NLP1. Input layer (max # of words are 32 in caption, shorter captions to be padded) as sequence of integers
NLP2. Map the integers with the vocabulary as vector
NLP3. Erase artifically some words in the caption to prevent overfitting
CNN1. Input Layer with 2048 features max
CNN2. Erase artifically some words in the caption to prevent overfitting
CNN3. Reduce (condense) features to 256
both4. Merge both inputs (add)
both5. Set ReLU activation function.
both6. (final)> vocab_size nodes and a softmax activation function, which outputs a probability distribution over the vocabulary for each word in the caption.

Use the categorical cross-entropy loss function and the Adam optimization algorithm.

    """


#    def __init__(self):
    
#    def forward(self):


# MODEL definition
# ================

frcnn_model = ResNet50Features      # Assign class ResNet50 to variable frcnn model

# the tokinzer is being initialized later (pre-trained or with a custom vocabulary)


# region FUNCTION definition
# ==========================

# Basic functions
# ---------------
def load_doc(in_pathname: str, in_filename: str):
    """
    The function load_doc takes in two arguments in_pathname and in_filename, both of which are strings. The
    in_pathname is the fully qualified path of the directory where the file is located, while in_filename is the
    name of the file in that directory.
    The function opens the file located at in_pathname/in_filename and reads its contents into memory as a string.
    The contents of the file are then returned by the function. If the file does not exist, the function will print
    an error message and exit the program.

    :param in_pathname: fully qualified path
    :type in_pathname: str
    :param in_filename: name of file in the directory
    :type in_filename: str
    :return: variable containing the content of the file, which was read
    :rtype: str
    """
    # Opening the file as read only
    full_path = os.path.join(in_pathname, in_filename)
#    print("[INFO] Path to file utilized: ", full_path)

    if not os.path.isfile(path=full_path):
        print(f"[INFO] File {in_filename} successfully opened in {in_pathname}.")
        exit()

    with open(full_path, "r") as file:
        text = file.read()

    return text

def merge_two_dicts(in_dict_one: dict, in_dict_two: dict):
    """
    Given two dictionaries, merge them into a new dict as a shallow copy.
    
    :param in_dict_one: dictionary one to be merged
    :type in_dict_one: dict
    :param in_dict_two: dictionary two to be merged
    :type in_dict_two: dict

    :return: dictionary (merged)
    :rtype: dict
    """
    merged_dict = in_dict_one.copy()
    merged_dict.update(in_dict_two)
    return merged_dict

def print_dict_with_list(in_dictionary: dict):
    """
    The code is a Python function that takes a dictionary d as input and prints its contents. The dictionary is
    assumed to contain lists as values.

    1. The function first loops through the items in the dictionary using the items method. For each key-value pair,
    it prints the key and a colon followed by a newline.

    2. Next, it uses another loop to iterate over the items in the list associated with the key. The inner loop prints
    each item in the list on a separate line.

    The function does not return any output, it only prints the contents of the dictionary.

    :param in_dictionary: dictionary containing lists as values
    :type in_dictionary: dict
    """
    for key, value in in_dictionary.items():
        print(key, ":")
        for item in value:
            print(item)


# Vectorizing functions
# ---------------------

# ToDo Stefan: outsource / transfer into separate script in folder data
def caption_description_file(in_path_2_text: str,
                             in_flickr_caption_file: str,
                             in_path_2_out: str,
                             out_file_w_clean_captions: str):
    
    # ToDo SWM: Besser die Funktion in ein packages auslagern. Nomenklatur: _ in Funktionsnamen als Indikation fuer
    #  "private" Funktion
    def generate_image_dict(in_path_text_data: str,
                            in_description_filename: str):
        """
        Generate description_dictionary for look up of possible captions (if more than one)

            read line
            find #
            treat everything before # as key
            treat everything after the position of # +2 as caption
            check if key exists and
            if exists, add caption
            if not exists, add key + caption
            clean sentences from \n and \t

        {"image_name1":["caption1", "caption2", "caption3"], "image_name2":...}
        :param in_path_text_data: path to raw data containing text files
        :type in_path_text_data: str
        :param in_description_filename: name of text file containing each image name and its captions, here. Flickr 8k file
        :type in_description_filename: str

        :return: dictionary containing for each image (=key) all the captions as values in a list
        :rtype: dict
        """
        
        caption_dict = {}
        
        image_description_list = load_doc(in_pathname=in_path_text_data, in_filename=in_description_filename)
        
        for aline in image_description_list.split(
                "\n"):  # reads the file line by line using \n as indicator for new line
            
            # check if search-string is present on a current line
            if aline.find(search_string) != -1:
                pos_string = aline.find(search_string)
                
                key = aline[0:pos_string]
                value = aline[pos_string + 3:].rstrip("\n\t")
                if key in caption_dict:
                    caption_dict[key].append(value)
                else:
                    caption_dict[key] = [value]
        
        print("[RUNNING] Dictionary with image names and their captions generated.")
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
                list_value[item] = re.sub("\s{2,}", " ", list_value[item]).strip()  # delete trailing and leading spaces
                # as well as multiple spaces
            
            # 3. delete all values from first key
            # 4. Write back cleaned list to first key
            in_dict[keys] = list_value
        
        # 5. Go to next key
        print("[RUNNING] Clean dictionary generated; removal of punctuation and 'a, A'")
        
        return in_dict
    
    def save_descriptions(in_clean_dict: dict,
                          in_path_output: str,
                          in_filename_output: str):
        """
        This function will create a list of all the descriptions that have been preprocessed and store them into a file.

        :param in_clean_dict:
        :type in_clean_dict:
        :return:
        :rtype:
        """
        
        full_path = os.path.join(in_path_output, in_filename_output)

        with open(full_path, "w") as f:
            for i, (key, values) in enumerate(in_clean_dict.items()):
                for value in values:
                    f.write("{}: {}".format(key, value))
                    if i < len(in_clean_dict) - 1 or value != values[-1]:
                        f.write("\n")

        print(f"[RUNNING] File {in_filename_output} in path {in_path_output} generated.")
        
        return
    
    ### FUNCTION MAIN
    print("\n[START] Generating a text file containing image name and related caption.")
    description_dictionary = generate_image_dict(in_path_text_data=in_path_2_text,
                                                 in_description_filename=in_flickr_caption_file)
    
    clean_dictionary = clean_dict(in_dict=description_dictionary)
    
    #ToDo Stefan  open description.json  und dump file als json.dump (statt pickl) und damit kann die Funktion
    # eingespart werden
    save_descriptions(in_clean_dict=clean_dictionary,
                      in_path_output=in_path_2_out,
                      in_filename_output=out_file_w_clean_captions)

    print(f"[DONE]\tCaptions from Flickr source file extracted\n"
          f"\t\tCaptions cleaned from numbers, punctuation and counting like 'a'\n"
          f"\t\tDescription txt-file with image name and caption in one line saved in <{in_path_2_out}"
          f"{out_file_w_clean_captions}>\n")
    
    return


def extract_features_from_images(in_path_processed_data: str,
                                 in_file_image_features: str,
                                 in_path_image_data: str):
    """
   extract features from images
   a) check, if feature file exists and how many keys were saved
   b) if file exists, ask user about action (redo or skip)
    
    :param in_path_processed_data:
    :type in_path_processed_data:
    :param in_file_image_features:
    :type in_file_image_features:
    :return:
    :rtype:
    """

    def extract_features(in_data_path: str, in_path_processed_data: str, in_file_feature: str):
        """
        This code is for extracting features from a set of images. The images are read from the directory specified by
        in_data_path. The code uses a pre-trained ResNet50 model from PyTorch to extract features from each image. The images
        are first resized to have a height or width of 224 pixels, then converted to a tensor format, and then passed
        through the ResNet50 model to obtain features. The features and the corresponding image file names are stored in
        a dictionary and saved in a pickle file specified by in_file_feature. The code saves the features in the pickle
        file after processing every 100 images, and merges the current feature dictionary with the previously saved
        dictionary. At the end, the final feature dictionary is saved in the same pickle file.

        :param in_file_feature: file, where the extracted features were saved so far
        :type in_file_feature: str
        :param in_path_processed_data: root path for all processed data
        :type in_path_processed_data: str
        :param in_data_path: Path to directory containing all image files
        :type in_data_path:
        :return:
        :rtype:
        """
        feature_dict = {}
        image_list = os.listdir(in_data_path)  # generate a list of images
        resnet50_features = ResNet50Features()
    
        print("\n[INFO] Starting to work on feature extraction (might take some time!!)...")
    
        for n in range(len(image_list)):
        
            filename = in_data_path + "/" + image_list[n]
            image = cv2.imread(filename)  # height, width, color
        
            # 2. transform image (resize, normalize, convert to Tensor, reshape (incl. batch size))
            height, width = image.shape[0], image.shape[1]
        
            if height > width:
                reduction_ratio = 224 / height
                # dimension for resize needs to be width, height
                dim = [int(image.shape[1] * reduction_ratio), int(image.shape[0] * reduction_ratio)]
            else:
                reduction_ratio = 224 / width
                dim = [int(image.shape[1] * reduction_ratio), int(image.shape[0] * reduction_ratio)]
        
            img_resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
            
            # ToDo Stefan: Muessen die Bilder ggf. gepadded werden, damit die Bilder 224x224 einheitlich haben? Dann
            #  koennen die Bilder auch gebatched werden.
            
            # 3. convert to tensor
            convert_tensor = transforms.ToTensor()
            img_pytorch = convert_tensor(img_resized)
            img_pytorch = img_pytorch.reshape(1, img_pytorch.shape[0], img_pytorch.shape[1], img_pytorch.shape[2])
        
            # 2a. normalize?: transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        
            features = resnet50_features(img_pytorch)
        
            # 3 map image name with feature matrix, i.e. establish dictionary with image name and feature
            feature_dict[image_list[n]] = features
        
            if (n + 1) % 100 == 0:
            
                # check, if feature file already exists
                if os.path.isfile(path=os.path.join(in_path_processed_data, in_file_feature)):
                    # load pickle file with all features in dict saved so far
                    all_features = pickle.load(open(os.path.join(in_path_processed_data, in_file_feature), "rb"))
                
                    # add new feature dict to all_feature dict
                    feature_dict = merge_two_dicts(in_dict_one=all_features, in_dict_two=feature_dict)
                
                    # save feature dict to file system
                    pickle.dump(feature_dict, open(os.path.join(in_path_processed_data, in_file_feature), "wb"))
                else:
                    # save feature dict to file system
                    pickle.dump(feature_dict, open(os.path.join(in_path_processed_data, in_file_feature), "wb"))
            
                # reset feature dict to 0
                feature_dict = {}
            
                print(
                    f"[Processing...] First {n + 1} images done! Let's go for the remaining {len(image_list) - n} images!")
    
        # 4 save everything into file *dump as pickle
    
        # load pickle file with all features in dict saved so far
        all_features = pickle.load(open(os.path.join(in_path_processed_data, in_file_feature), "rb"))
    
        # add new feature dict to all_feature dict
        feature_dict = merge_two_dicts(in_dict_one=all_features, in_dict_two=feature_dict)
    
        # save feature dict to file system
        pickle.dump(feature_dict, open(os.path.join(in_path_processed_data, in_file_feature), "wb"))
    
        print(f"\n[INFO] Finally done! AND ... I saved my doing in {in_file_feature} for you!")
    
        return
 
    # FUNCTION main
    print("\n[INFO] Checking availability of feature extraction file generated previously.")
    feature_path = os.path.join(in_path_processed_data, in_file_image_features)
    
    if os.path.isfile(path=feature_path):
        feature_dict = pickle.load(open(os.path.join(in_path_processed_data,
                                                     in_file_image_features), "rb"))
        
        print(f"[ATTENTION] I just discovered, that I already generated once a feature file with significant "
              f"efforts!\n"
              f"\t\t\tWe talk here about {len(feature_dict)} entries!!!\n"
              f"\t\t\tYou have now the one chance to skip the generation of a new feature dict!")
        
        user_decision = input("\n[INPUT] Do you want to skip the generation of a new feature dictionary (y/n): "
                              "").lower()
        
        if user_decision == "n":
            extract_features(in_data_path=in_path_image_data,
                             in_path_processed_data=in_path_processed_data,
                             in_file_feature=in_file_image_features)
        else:
            print(
                "[INFO] You made the wise decision to save me and you time by not generating a new feature dictionary.")
    else:
        extract_features(in_data_path=in_path_image_data,
                         in_path_processed_data=in_path_processed_data,
                         in_file_feature=in_file_image_features)
    
    return

'''
probably obsolete
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
    print("[INFO] List of unique words (out of captions) generated.\n")
    return list_of_words
'''

def get_image_list(in_path_txt_data: str, in_file_train_img_lst: str):
    """
    The function starts by creating an empty list called image_lst that will store the image file names.

    It then prints a message indicating that it is processing the text input file and calls another function load_doc
    with in_pathname and in_filename as arguments. This function is loading the contents of the input text file into
    memory.

    The contents of the input text file are then split by newline characters ("\n") and stored in the image_lst list.
    The split operation discards the last element in the list if it is an empty string.

    Finally, the function returns the image_lst list.

    :param in_path_txt_data: A string representing the path to the directory containing the input text file
    :type in_path_txt_data:str
    :param in_file_train_img_lst: A string representing the name of the input text file that contains the list of
    image file names
    :type in_file_train_img_lst: str
    :return: list of image names
    :rtype: list
    """
    image_lst = []
    
    print("\n[INFO] Processing text input file: ", in_file_train_img_lst)
    image_lst = load_doc(in_pathname=in_path_txt_data, in_filename=in_file_train_img_lst)
    image_lst = image_lst.split("\n")[:-1]
    print("[INFO] List with image file-names generated.")

    return image_lst


def generate_dic_with_tokens(in_path_processed_data: str, in_file_image_description: str, in_image_list):
    """
    This function will create a dictionary that contains captions for each photo from the list of photos. We also append
    the "start" and "end" identifier for each caption. We need this so that our LSTM model can identify the starting and
    ending of the caption.
    :param in_path_processed_data: path to description file on disk
    :type in_path_processed_data: str
    :param in_file_image_description: name of file on disk
    :type in_file_image_description: str
    :param in_image_list: name of list containing all image names
    :type in_image_list: list
    """
    # import description file with image name and caption on one line
    description_file = load_doc(in_path_processed_data, in_file_image_description)
    descriptions = {}
    
    for line in description_file.split("\n"):   # split text file line by line

        # split the line into single words
        words = line.split()

        if len(words) < 1:
            continue
        # assign the first word as image name and all other words as image caption
        image, image_caption = words[0][:-1], words[1:]

        # check, if image name exists in the list of images provided by flickr in the raw data
        if image in in_image_list:
            
            if image not in descriptions:
                # if the image name does not exist in the descriptions dictionary
                descriptions[image] = []            # generate an empty list

                # KERAS: generate a string consisting of the image caption and start + end
                # desc = '<start> ' + " ".join(image_caption) + ' <end>'

                # HuggingFaces: generate a string consisting of the image caption and start + end
                desc = '<CLS> ' + " ".join(image_caption) + ' <SEP>'

                # add a new dictionary entry with the image name as key and the caption + start/end as value
                descriptions[image].append(desc)
            else:
                # generate a string consisting of the image caption and start + end

                # KERAS: generate a string consisting of the image caption and start + end
                # desc = '<start> ' + " ".join(image_caption) + ' <end>'

                # HuggingFaces: generate a string consisting of the image caption and start + end
                desc = '<CLS> ' + " ".join(image_caption) + ' <SEP>'

                descriptions[image].append(desc)
    
    print(f"\n[INFO] Dictionary with cleaned captions and tokens start and end generated.\n")
    return descriptions


def load_features(in_path_processed_data: str, in_file_saved_features: str, in_images_to_train: list):

    # loading all features
    all_features = pickle.load(open(os.path.join(in_path_processed_data,in_file_saved_features), "rb"))

    # selecting only needed features
    features_list = {k: all_features[k] for k in in_images_to_train}
    
    return features_list

    
def dict_to_list(descriptions: dict):
    """
    The function dict_to_list takes a dictionary descriptions as input and returns a list of all the values of the
    dictionary.
    
    1. The function initializes an empty list all_desc to store the values. Then, it loops through the keys of the
    input dictionary descriptions using the keys method.
    
    2. For each key in the dictionary, the function uses a list comprehension to extract the values associated with
    that key and appends each of these values to the all_desc list.
    
    3. Finally, the function returns the resulting list all_desc which contains all the values from the input
    dictionary.
    
    :param descriptions: Dictionary with lists as values for each key.
    :type descriptions: dict
    :return: list of all descriptions withput reference to imqge
    :rtype: list
    """
    
    all_desc = []
    for key in descriptions.keys():
        [all_desc.append(d) for d in descriptions[key]]
    return all_desc


def generate_list_with_unique_words(in_path_input_path : str, in_description_file_input : str):
    """
    Read a cleaned text file with image name in the first column and followed by possible captions.
    Generate a list out of the input containing the words used in the captions
    1. load the file from harddrive
    2. iterate through the file and read eachtime all words starting from index 1 (excluding the image name)
    3. append to a list checking, if a word already exists in the dictionary
    4. Add the specific tokens <start>, <end>,
    :param in_description_file_input: ext file containing image name and caption
    :type in_description_file_input: str
    :param in_path_input_path: path to input file containing descriptions
    :type in_path_input_path: str
    
    :return:
    :rtype:
    """
    list_of_words = []
 
    file_description = load_doc(in_pathname=in_path_input_path, in_filename=in_description_file_input)
#    print(file_description)

    for line in file_description.split("\n"):  # split text file line by line
    
        # split the line into single words
        words = line.split()
    
        if len(words) < 1:
            continue
        # assign the first word as image name and all other words as image caption
        image, image_caption = words[0], words[1:]
    
        for word in image_caption:
            if word not in list_of_words:
                list_of_words.append(word)
        
    return list_of_words



def max_length(in_path_input_path: str, in_description_file_input: str):
    """
    # calculate maximum length of descriptions
    :param descriptions:
    :type descriptions:
    :return:
    :rtype:
    """

    max_length = 0
    
    file_description = load_doc(in_pathname=in_path_input_path, in_filename=in_description_file_input)

    for line in file_description.split("\n"):  # split text file line by line
    
        # split the line into single words
        words = line.split()
    
        if len(words) < 1:
            continue
        # assign the first word as image name and all other words as image caption
        image, image_caption = words[0], words[1:]
        
        if len(image_caption) > max_length:
            max_length = len(image_caption)
            
    return max_length

# endregion

# ===================
# MAIN
# ===================
print("\n[START] Start of execution {}\n".format(time.strftime("%H:%M:%S")))

# generate a dictionary with all cleaned captions (i.e. w/o a, ...) for each file as base for later training
# ---------------------------------------------------------------------------------------------------------
caption_description_file(in_path_2_text= path_2_text_data,
                         in_flickr_caption_file= file_image_flickr_description,
                         in_path_2_out= path_2_processed_data,
                         out_file_w_clean_captions= file_image_clean_caption)
                        
# extract features from images after checking, if feature file already exists
extract_features_from_images(in_path_processed_data=path_2_processed_data,
                             in_file_image_features = file_image_features,
                             in_path_image_data=path_image_data)


# DATASET (for training the model)

# generate list of training images based on txt-file containing the image names used for training
train_images = get_image_list(in_path_txt_data=path_2_text_data,
                              in_file_train_img_lst=file_train_img_lst)

# generate dictionary including only the images (key) and caption (value) preceded by "start" and trailed by
# "end" we want to train (i.e. which are in the training list)
dict_train_descriptions = generate_dic_with_tokens(in_path_processed_data=path_2_processed_data,
                                                   in_file_image_description=file_image_clean_caption,
                                                   in_image_list=train_images)

# load features for selected training files
train_features = load_features(in_path_processed_data=path_2_processed_data,
                               in_file_saved_features=file_image_features,
                               in_images_to_train=train_images)


# 5. Tokenizing the vocabulary
# ----------------------------
"""
Computers don’t understand English words, for computers, we will have to represent them with numbers. So,
we will map each word of the vocabulary with a unique index value. Keras library provides us with the tokenizer
function that we will use to create tokens from our vocabulary and save them to a **“tokenizer.p”** pickle file.

An LSTM tokenizer is a type of deep learning model that uses Long Short-Term Memory (LSTM) units to tokenize text.
Here is an outline of how you can create an LSTM tokenizer in Python:

1. Preprocess the text data: The first step is to preprocess the text data. This includes converting the text to
lowercase, removing punctuation, and splitting the text into sentences and tokens. You can use regular expressions
and other text preprocessing tools to achieve this.

2. Create a vocabulary: Create a vocabulary of the words in the text data. This vocabulary can be used to encode the
text data into numerical representations, which can be fed into the LSTM model. You can use a word embedding to map
words to vectors and use a one-hot encoding to map words to numbers.

3. Define the LSTM model architecture: Define the architecture of the LSTM model. This involves specifying the number
of LSTM units, the number of layers, the activation function, and the input and output shapes.

4. Train the LSTM model: Train the LSTM model on the preprocessed text data. You can use a categorical cross-entropy
loss function and an optimizer such as Adam to train the model. You should also set aside a portion of the data for
validation to monitor the model's performance during training.

5. Tokenize the text data: Once the model is trained, you can use it to tokenize new text data. You can feed the
text data into the trained LSTM model and use the output to generate the tokenized text.

6. Save and load the model: Finally, save the trained LSTM model so that you can use it later without having to
train it again. You can use the Keras or TensorFlow libraries in Python to save and load the model.
"""

# Text to be tokenize
text_to_be_tokenized = dict_to_list(descriptions=dict_train_descriptions)

# generate list of words contained in captions
# ToDo Stefan:
"""
1. Alle Strings aus @text_to_be_tokenized verknuepfen
    all_strings = " ".join(text_to_be_tokenized)
2. split by \n = list of words
list_of_words = all_strings.split(" ")
3. "Set" eliminates alle doppelten Worte und generiert eine Liste
result = list(set(list_of_words))
result.remove ("<SEP>, ...")
"""

word_list = generate_list_with_unique_words(in_path_input_path=path_2_processed_data,
                                            in_description_file_input=file_image_clean_caption)

# save list of words to file to be used as input for tokenizer
full_voc_path = os.path.join(path_2_processed_data, file_vocabulary)
with open(full_voc_path, 'w') as file:
    for word in word_list:
        file.write(f'{word}\n')


# Initialize and train the tokenizer with the given sentences, i.e. assign each word a unique integer
# tokenizer = transformers.BertTokenizer(vocab_file=full_voc_path)
#Todo Sebastian: Warum funktioniert der eigene Tokenizer nicht?
#ToDo Sebastian: Warum erhalte ich ein Token mit dem Index 30522 am Anfang des Satzes ohne Bezug zum Satz
# 30522 und 30523 sind die special tokens, die ich manuell hinzugefuegt hatte. Fuer dieses Modell brauchen wir die
# manuell hinzugefuegten Tolens SEP und CLS in der cpation nicht! Funktion aendern!
tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased')
# tokenizer.add_tokens(special_tokens)
print("[INFO] Size of vocabulary for tokenizer", tokenizer.vocab_size)

# Tokenize the text
tokenized_sentences = [tokenizer.tokenize(sentence) for sentence in text_to_be_tokenized]

if any(val is None for val in tokenized_sentences):
    print("There is a None value in the tokenized sentences.")
    
# Convert the tokenized text to an encoding that the model can use
encoded_sentences =[]


try:
    #    encoded_sentences = [tokenizer.encode(sentence) for sentence in tokenized_sentences]
    for sentence in tokenized_sentences:
        # print(sentence)
        encoded_sentences.append(tokenizer.encode(sentence))
        # print(tokenizer.encode(sentence))

except KeyError as e:
    print(f"Encountered an error while encoding: {e}")


# Print the tokenized and encoded text
#print("Tokenized Text:", tokenized_sentences)
#print("Encoded Text:", encoded_sentences)


tokenizer.save_pretrained(os.path.join(path_2_processed_data, path_tokenizer))
# tokenizer = BertTokenizer.from_pretrained('./modified_tokenizer')

max_capt_length= max_length(in_path_input_path=path_2_processed_data,
                            in_description_file_input=file_image_clean_caption)

print("[INFO] Max. length of caption: ", max_capt_length)




"""# Define the data generator
def data_generator(descriptions, features, tokenizer, max_length):
    for key, description_list in descriptions.items():
        # Retrieve photo features
        feature = features[key][0]

        for desc in description_list:
            # Tokenize the description
            input_ids = tokenizer.encode(desc, return_tensors="pt")
            # Pad the input sequence if it's shorter than `max_length`
            input_ids = input_ids.padding(max_length, value=tokenizer.pad_token_id)
            # Add the photo feature to the input
            input_ids = torch.cat((feature, input_ids), dim=0)
            # Generate the target token
            target_ids = tokenizer.encode(desc[1:], return_tensors="pt")
            # Convert the target token to a categorical format
            target_ids = tokenizer.to_categorical(target_ids, num_classes=tokenizer.vocab_size)
            yield [input_ids, target_ids]
          
            
[a,b],c = next(data_generator(dict_train_descriptions, train_features, tokenizer, max_capt_length))
print(a.shape, b.shape, c.shape)
"""
print("\n[END] End of execution: ", time.strftime("%H:%M:%S"))