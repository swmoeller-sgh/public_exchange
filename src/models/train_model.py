"""
Objective
==============
Generate the caption for an image using two NN (CNN and NLP).
https://data-flair.training/blogs/python-based-project-image-caption-generator-cnn/
"""


# Library import
# ===============
import torchvision.models
from torchvision import transforms

import torch.nn

import os
import re
import time
import cv2

import pickle

import PIL

import numpy as np

import torch

# model = models.vgg16(pretrained=True)

# Variable definition
# ===================

# INPUT data
path_text_data = "/Users/swmoeller/python/prj_image_captioning_e2e/data/05_raw/Flickr8k_text"
file_image_flickr_description = "Flickr8k.token.txt"
file_train_img_lst = "Flickr_8k.trainImages.txt"


search_string = "#"         # string to be found in text file to identify captions

# TRAINING directory
path_train_dir = "/Users/swmoeller/python/prj_image_captioning_e2e/data/05_raw/Flicker8k_Dataset"

# OUTPUT data
path_processed_data = "/Users/swmoeller/python/prj_image_captioning_e2e/data/20_processed/"
file_image_description = "description.txt"
file_image_features = "feature.p"

# CLASS definition
# ================


# FUNCTION definition
# ===================

def load_doc(in_pathname: str, in_filename: str):
    """
    Loads a textfile from the directory structure into the memory

    :param in_pathname: fully qualified path
    :type in_pathname: str
    :param in_filename: name of file in the directory
    :type in_filename: str
    :return: variable containing the content of the file, which was read
    :rtype: str
    """
    # Opening the file as read only
    full_path = os.path.join(in_pathname, in_filename)
    print("[INFO] Complete path to file now being opened: ", full_path)

    if os.path.isfile(path=full_path):
        file = open(file=full_path, mode="r")
        text = file.read()
        file.close()
        print(f"[INFO] File {in_filename} successfully opened in {in_pathname}\n")
    else:
        print(f"\n{in_pathname} or {in_filename} does not exist!\nThat sucks, I quit my service!")
        exit()
    
    return text


def generate_image_dict(in_path_text_data: str, in_description_filename: str):
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
    :param in_description_filename: name of text file containing each image name and its captions
    :type in_description_filename: str
    
    :return: dictionary containing for each image (=key) all the captions as values in a list
    :rtype: dict
    """

    caption_dict = {}
    
    image_description_list = load_doc(in_pathname=in_path_text_data, in_filename=in_description_filename )
    
    for aline in image_description_list.split("\n"): # reads the file line by line using \n as indicator for new line

        # check if search-string is present on a current line
        if aline.find(search_string) != -1:
            pos_string = aline.find(search_string)
            
            key = aline[0:pos_string]
            value = aline[pos_string+3:].rstrip("\n\t")
            if key in caption_dict:
                caption_dict[key].append(value)
            else:
                caption_dict[key] = [value]

    print("[INFO] Dictionary with image names and their captions generated.\n")
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
    print("[INFO] Clean dictionary generated; removal of punctuation and 'a, A'\n")
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
    print("[INFO] List of unique words (out of captions) generated.\n")
    return list_of_words


def save_descriptions(in_clean_dict: dict, in_path_output : str, in_filename_output : str):
    """
    This function will create a list of all the descriptions that have been preprocessed and store them into a file.
    
    :param in_clean_dict:
    :type in_clean_dict:
    :return:
    :rtype:
    """
    full_path = os.path.join(in_path_output, in_filename_output)
    outfile = open(full_path, "w")
    # output the header row
    outfile.write("image\tcaption\n")
    # output image name and each caption line-by-line
    for keys, values in in_clean_dict.items():
        for items in range(len(values)):
            row_string = "{}\t{}\n".format(keys, values[items])
            outfile.write(row_string)
    
    outfile.close()
    print(f"[INFO] File {in_filename_output} in path {in_path_output} generated.")

    return

"""
def setup_extraction_model(in_image):
    
    return"""

def read_image_list(in_root_path_txt_data: str, in_train_img_lst: str):
    image_lst = []
    
    list_images = os.path.join(in_root_path_txt_data, in_train_img_lst)
    print("[INFO] Processing text input file: ", in_train_img_lst)
    if os.path.isfile(path=list_images):
        image_list_file = open(file=list_images, mode="r")
        image_lst = image_list_file.read()
        image_lst = image_lst.split("\n")[:-1]
    else:
        print(f"{in_root_path_txt_data} or {in_train_img_lst} do not exist!")
        exit()

    print("[INFO] List with image file-names generated.")

    return image_lst


def generate_clean_descriptions(in_path_data_text: str, in_file_image_description: str, in_image_list):
    """
    This function will create a dictionary that contains captions for each photo from the list of photos. We also append
    the "start" and "end" identifier for each caption. We need this so that our LSTM model can identify the starting and
    ending of the caption.
    :param in_path_data_text: path to description file on disk
    :type in_path_data_text: str
    :param in_file_image_description: name of file on disk
    :type in_file_image_description: str
    :param in_image_list: name of list containing all image names
    :type in_image_list: list
    """
    # import description file with image name and caption on one line
    description_file = load_doc(in_path_data_text, in_file_image_description)
    descriptions = {}
    
    for line in description_file.split("\n"):   # split text file line by line

        # split the line into single words
        words = line.split()

        if len(words) < 1:
            continue
        # assign the first word as image name and all other words as image caption
        image, image_caption = words[0], words[1:]

        # check, if image name exists in the list of images provided by flickr in the raw data
        if image in in_image_list:
            
            # if the image name does not exists in the descriptions dictionary
            if image not in descriptions:
                descriptions[image] = []            # generate an empty list
                desc = '<start> ' + " ".join(image_caption) + ' <end>'  # generate a string consisting of the image
                # caption and start + end
                # add a new dictionary entry with the image name as key and the caption + start/end as value
                descriptions[image].append(desc)
    
    print(f"[INFO] Dictionary with cleaned captions and tokens start and end generated.\n")
    
    return descriptions


def load_features(in_path_processed_data: str, in_file_saved_features: str, in_images_to_train: list):

    # loading all features
    all_features = pickle.load(open(os.path.join(in_path_processed_data,in_file_saved_features), "rb"))
    # selecting only needed features
    features_list = {k: all_features[k] for k in in_images_to_train}
    
    return features_list
    

# MODEL definition
# ================

#OPTION 1
frcnn_model = torchvision.models.get_model("resnet50", weights="DEFAULT")

#  replace fc with Identity, which just returns the input as the output, and since the features are its input,
#  the output of the entire model will be the features.
frcnn_model.fc = torch.nn.Identity()
# print("option 1", frcnn_model)


# Option 2

class FeatureExtractor(torch.nn.Module):
    def __init__(self, model):
        super(FeatureExtractor, self).__init__()
        
        self.model = model
#        print("\n[INFO] Original CNN-model\n", self.model)
        
        # replace feature layer
        self.model.fc = torch.nn.Identity()

        # Convert the image into one-dimensional vector
        self.model.flatten = torch.nn.Flatten()
        
#        print("\n\n[UPDATE] Modified CNN-model\n", self.model)
    
    def forward(self, x):
        # It will take the input 'x' until it returns the feature vector called 'out'
        out = self.model(x)
        out = self.fc(out)
        out = self.flatten(out)
        return out

# Initialize the model
model = torchvision.models.get_model("resnet50", weights="DEFAULT")
new_model = FeatureExtractor(model)


# ===================
# MAIN
# ===================
print("\n[START] Start of execution {}\n".format(time.strftime("%H:%M:%S")))
description_dictionary = generate_image_dict(in_path_text_data=path_text_data,
                                             in_description_filename=file_image_flickr_description)

clean_dictionary = clean_dict(description_dictionary)
unique_words = unique_vocabulary(clean_dictionary)
save_descriptions(in_clean_dict=clean_dictionary,
                  in_path_output=path_processed_data,
                  in_filename_output= file_image_description)


"""
https://www.projectpro.io/recipes/use-resnet-for-image-classification-pytorch
https://towardsdatascience.com/image-feature-extraction-using-pytorch-e3b327c3607a
"""

# 1. Load the image

img = cv2.imread(r"/Users/swmoeller/python/prj_image_captioning_e2e/data/05_raw/Flicker8k_Dataset"
                     r"/3423802527_94bd2b23b0.jpg") # height, width, color
print(f"\n[INFO] Current image parameter: Type of image is {type(img)} and its shape (height, width, color) is"
      f" {img.shape}.\nNow transforming...!")

# 2. transform image (resize, normalize, convert to Tensor, reshape (incl. batch size))
height = img.shape[0]
width = img.shape[1]
if height > width:
    reduction_ratio = 224/height
    # dimension for resize needs to be width, height
    dim=[int(img.shape[1]*reduction_ratio), int(img.shape[0]*reduction_ratio)]
    print(dim)
else:
    reduction_ratio = 224/width
    dim = [int(img.shape[1]*reduction_ratio), int(img.shape[0]*reduction_ratio)]
    print(dim)
img_resized = cv2.resize(img, dim,interpolation = cv2.INTER_AREA)
convert_tensor = transforms.ToTensor()
img_pytorch = convert_tensor(img_resized)
img_pytorch = img_pytorch.reshape(1, img_pytorch.shape[0],img_pytorch.shape[1],img_pytorch.shape[2])
print(f"\n[INFO] New image parameter: Type of image is {type(img_pytorch)} and its shape (batch, color, height, "
      f"width) is {img_pytorch.shape}.")

# 2a. normalize?: transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
# 3. turn of gradient (for what do we need gradient normally?)

# 4. put model to evaluation (with pre-trained weights)
frcnn_model.eval()
features = frcnn_model(img_pytorch)

# does not work :-(
#new_model.eval()
#features = new_model(img_pytorch)

# print("[INFO] Parameter of selected model for image recognition", frcnn_model)
print(features)


# 3 map image name with feature matrix, i.e. establish dictionary with image name and feature
feature_dict = {}

feature_dict["17273391_55cfc7d3d4.jpg"] = features
print(feature_dict)


# 4 save everything into file *dump as pickle

pickle.dump(features, open(os.path.join(path_processed_data, file_image_features),"wb"))



# Loading dataset for Training the model
# In the Flickr\_8k\_test folder, we have Flickr\_8k.trainImages.txt file that contains a list of 6000 image
# names that we will use for training.

# For loading the training dataset, we need more functions:
# - load\_photos( filename ) – This will load the text file in a string and will return the list of image names.
# - load\_clean\_descriptions( filename, photos ) – This function will create a dictionary that contains captions
# for each photo from the list of photos. We also append the "start" and "end" identifier for each caption. We need this so that our LSTM model can identify the starting and ending of the caption.
# - load\_features(photos) This function will give us the dictionary for image names and their feature vector
# which we have previously extracted.

# read names of images into a list
train_images = read_image_list(in_root_path_txt_data=path_text_data, in_train_img_lst=file_train_img_lst)

# generate dictionary of image (key) and caption (value) preceded by "start" and trailed by "end"
train_descriptions = generate_clean_descriptions(in_path_data_text=path_processed_data,
                                                 in_file_image_description=file_image_description,
                                                 in_image_list=train_images)
# load features for selected training files
train_features = load_features(in_path_processed_data= path_processed_data,
                               in_file_saved_features= file_image_features,
                               in_images_to_train=train_images)

print("\n[END] End of execution: ", time.strftime("%H:%M:%S"))














"""

# Loading data
test_transforms = transforms.Compose([transforms.Resize((224, 224)),
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                      ])
test_data = torchvision.datasets.ImageFolder(train_dir, transform= test_transforms)

test_loader = torch.utils.data.Dataloader(test_data, num_workers=0, batch_size=128)"""


"""img = PIL.Image.open(r"/Users/swmoeller/python/prj_image_captioning_e2e/data/05_raw/Flicker8k_Dataset"
                     r"/17273391_55cfc7d3d4.jpg")
img = img.resize((224, 224))
img.reshape(1, 3, 224, 224)

convert_tensor = transforms.ToTensor()
img_tensor = convert_tensor(img)

img_tensor= img_tensor.to(device)

frcnn_model.eval()
features = frcnn_model(img_tensor)
#features = new_model(img)

# Transform the image, so it becomes readable with the model
transform = transforms.Compose([
  transforms.ToPILImage(),
  transforms.CenterCrop(512),
  transforms.Resize(224),
  transforms.ToTensor()
])

# Will contain the feature
features = []

# Read the file
img = cv2.imread("Users/swmoeller/python/prj_image_captioning_e2e/data/05_raw/Flicker8k_Dataset/17273391_55cfc7d3d4.jpg")

# Transform the image
img = transform(img)
# Reshape the image. PyTorch model reads 4-dimensional tensor
# [batch_size, channels, width, height]
img = img.reshape(1, 3, 224, 224)
img = img.to("cpu")
# We only extract features, so we don't need gradient
with torch.no_grad():
# Extract the feature from the image
    feature= frcnn_model(img)
# Convert to NumPy Array, Reshape it, and save it to features variable
    features.append(feature.cpu().detach().numpy().reshape(-1))

# Convert to NumPy Array
features = np.array(features)
"""

# 1 define extractor class
# 2 Define extractor function
# 3 map image name with feature matrix
# 4 save everything into file *dump as pcikle