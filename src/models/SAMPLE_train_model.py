"""
Objective:

1. Preprocess the output (ground truth annotations/captions) so that each unique word is represented by a unique ID.

2. Given that the output sentences can be of any length, let's assign a start and end token so that the model knows
when to stop generating predictions. Furthermore, ensure that all input sentences are padded so that all inputs have
the same length.

3. Pass the input image through a pre-trained model, such as     VGG16, ResNet-18, and so on, to fetch features prior to
the flattening layer.

4. Use the feature map of the image in conjunction with the text obtained in the previous step (the start token if it
is the first word that we are predicting) to predict a word.

5. Repeat the preceding step until we obtain the end token.

"""

# region Library import
# ===============

# Makes sure that the directory for my individual scripts is in the PYTHONPATH.
import sys
sys.path.append('/Users/swmoeller/python/prj_image_captioning_e2e/src/util')

# my own little helpers
import common_dataframe_utils
import common_tokenizer_utils

# from torch_snippets import *
import json

import numpy as np
import pandas as pd

import torch

import os
import time

import torchtext

from torch.nn.utils.rnn import pack_padded_sequence
from torchvision import models

from pycocotools.coco import COCO
from collections import defaultdict

# Tools for downloading images and corresponding annotations from Google's OpenImages dataset.
from openimages.download import _download_images_by_id

from torchvision import transforms

from torchsummary import summary

from torch_snippets import *
"""
"""
# endregion


# region Variable definition
# ===================

# INPUT data
# ----------
IN_image_json = "/Users/swmoeller/python/prj_image_captioning_e2e/data/05_raw/google_image/open_images_train_v6_captions.jsonl"
max_download_images = 50

# OUTPUT data
# -----------
OUT_image_train_not_train_csv = "/Users/swmoeller/python/prj_image_captioning_e2e/data/20_processed/google_image/data" \
                                ".csv"
share_of_validation_images = 0.1        # Share of images to be marked for validation (thus excluded from training)

OUT_PATH_train_images = "/Users/swmoeller/python/prj_image_captioning_e2e/data/20_processed/google_image/train-images"
OUT_PATH_validation_images = "/Users/swmoeller/python/prj_image_captioning_e2e/data/20_processed/google_image/val-images"

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# endregion


# region CLASS definition
# ================
class Vocab:
    """
    This code defines an empty class called Vocab with no attributes or methods as a starting point to later add
    functionality to it as needed.
    """
    pass

class CaptioningData(Dataset):
    def __init__(self, root: str, df: pd.core.frame.DataFrame, vocab: object):
        """
        The __init__ method initializes the dataset object and takes in three arguments: root, df, and vocab.
        The reset_index(drop=True) method is called on df to reset the index and drop the old index column.
        The transform attribute is a Compose object from PyTorch's transforms module that applies a sequence of image
        transformations to the input images. In this case, the images are resized to 224x224 pixels, randomly cropped
        to a 224x224 square, randomly flipped horizontally, converted to a tensor, and normalized using the specified
        mean and standard deviation values.
        
        :param root: root is a string specifying the root directory of the dataset,
        :type root: str
        :param df: df is a Pandas DataFrame containing the image filenames and captions
        :type df: pd.core.frame.DataFrame
        :param vocab: Vocabulary object that maps words to their corresponding indices
        :type vocab: object
        """
        self.df = df.reset_index(drop=True)
        self.root = root
        self.vocab = vocab
        self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                                 (0.229, 0.224, 0.225))]
        )

    def __getitem__(self, index):
        """Returns one data pair (image and caption)."""
        
        row = self.df.iloc[index].squeeze()
        id = row.image_id
        image_path = f'{self.root}/{id}.jpg'
        image = Image.open(os.path.join(image_path)).convert('RGB')

        caption = row.caption
        tokens = str(caption).lower().split()
        # ToDo SWM Das haben wir doch schon gemacht! Hier werden die punctuations, etc. nicht bereinigt! Damit werden
        #  jetzt einige Worte nicht gefunden werden!
        
#        tokens = common_tokenizer_utils.get_clean_list_of_words(caption)    # replacement to above
        
        target = []
        target.append(vocab.stoi['<start>'])
        target.extend([vocab.stoi[token] for token in tokens])
        target.append(vocab.stoi['<end>'])
        target = torch.Tensor(target).long()
        return image, target, caption
    
    def choose(self):
        return self[np.random.randint(len(self))]
    
    def __len__(self):
        return len(self.df)
    
    def collate_fn(self, data):
        data.sort(key=lambda x: len(x[1]), reverse=True)
        images, targets, captions = zip(*data)
        images = torch.stack([self.transform(image) for image in images], 0)
        lengths = [len(tar) for tar in targets]
        _targets = torch.zeros(len(captions), max(lengths)).long()
        for i, tar in enumerate(targets):
            end = lengths[i]
            _targets[i, :end] = tar[:end]
        return images.to(device), _targets.to(device), torch.tensor(lengths).long().to(device)

class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        """Load the pretrained ResNet-152 and replace top fc layer."""
        super(EncoderCNN, self).__init__()
        resnet = models.resnet152(pretrained=True)
        modules = list(resnet.children())[:-1]  # delete the last fc layer.
        self.resnet = nn.Sequential(*modules)
        self.linear = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)
    
    def forward(self, images):
        """Extract feature vectors from input images."""
        with torch.no_grad():
            features = self.resnet(images)
        features = features.reshape(features.size(0), -1)
        features = self.bn(self.linear(features))
        return features


class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers, max_seq_length=80):
        """Set the hyper-parameters and build the layers."""
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.max_seq_length = max_seq_length
    
    def forward(self, features, captions, lengths):
        """Decode image feature vectors and generates captions."""
        embeddings = self.embed(captions)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        packed = pack_padded_sequence(embeddings, lengths.cpu(), batch_first=True)
        outputs, _ = self.lstm(packed)
        outputs = self.linear(outputs[0])
        return outputs
    
    def predict(self, features, states=None):
        """Generate captions for given image features using greedy search."""
        sampled_ids = []
        inputs = features.unsqueeze(1)
        for i in range(self.max_seq_length):
            hiddens, states = self.lstm(inputs, states)  # hiddens: (batch_size, 1, hidden_size)
            outputs = self.linear(hiddens.squeeze(1))  # outputs: (batch_size, vocab_size)
            _, predicted = outputs.max(1)  # predicted: (batch_size)
            sampled_ids.append(predicted)
            inputs = self.embed(predicted)  # inputs: (batch_size, embed_size)
            inputs = inputs.unsqueeze(1)  # inputs: (batch_size, 1, embed_size)
        
        sampled_ids = torch.stack(sampled_ids, 1)  # sampled_ids: (batch_size, max_seq_length)
        # convert predicted tokens to strings
        sentences = []
        for sampled_id in sampled_ids:
            sampled_id = sampled_id.cpu().numpy()
            sampled_caption = []
            for word_id in sampled_id:
                word = vocab.itos[word_id]
                sampled_caption.append(word)
                if word == '<end>':
                    break
            sentence = ' '.join(sampled_caption)
            sentences.append(sentence)
        return sentences
    

# endregion


# region FUNCTION definition
# ==========================
# region @FUNCTION CHECKED
def download_images(IN_image_json_file: str, IN_max_amount_of_images: int, IN_validation_share: float):
    """
Generates a training and validation dataset using a json file containing image name, caption and other information.

1. Loads the json file
2. Randomly picks the requested number of images from the json file and safes the information about each image as
dataframe
3. Inserts another column into the dataframe and classifies each image as training or validation image in this column
4. Processes the dataframe and downloads the images into the two folders for training and validation

    :param IN_image_json_file: json file including the dictionary with image names, captions, etc.
    :type IN_image_json_file: str
    :param IN_max_amount_of_images: Maximum amount of images to be downloaded
    :type IN_max_amount_of_images: int
    :param IN_validation_share: Percentage (in float) of images kept as validation images (this not included in
    training)
    :type IN_validation_share: float
    :return: pandas dataframe containing the randomly selected images with their captions, etc.
    :rtype: dataframe
    """
    
    # Fetch the dataset from the Open Images dataset
    with open(IN_image_json_file, 'r') as json_file:
        json_list = json_file.read().split('\n')
    np.random.shuffle(json_list)                            # avoid always downloading the same images
    print(f"[Info] JSON file {IN_image_json_file} loaded and shuffled!")

    # Generate a list of dataframes containing image name, caption and other information as column names and listing the
    # images and their details in each line
    data_df_list = []        # list containing later the single dataframes
    for index_no, json_str in enumerate(json_list):
        if index_no == IN_max_amount_of_images:
            break
        try:
            result = json.loads(json_str)
            
            """
            Create a DataFrame from a dictionary "result" with orient='index' argument to use the keys of the dictionary
            as the column names of the DataFrame.
            The .T attribute is then used to transpose the resulting DataFrame. The transpose operation switches the
            rows and columns of the DataFrame.
            """
            x = pd.DataFrame.from_dict(result, orient='index').T
            data_df_list.append(x)
        except:
            pass
    
    # Generate one dataframe out of the list of single datafranes and split the dataframe (data) into training and
    # validation data (i.e. mark the images as validation or training image)
    np.random.seed(10)
    data_df = pd.concat(data_df_list)
    # generate a list of all images with one image and its caption in one line plus an indication (true, false),
    # if used as training image
    data_df['train'] = np.random.choice([True, False], size=len(data_df), p=[1-IN_validation_share, IN_validation_share])

    # save the dataframe as csv file (without index)
    data_df.to_csv(OUT_image_train_not_train_csv, index=False)
    
    # Download all images marked in column "train" as "True" corresponding to the image IDs fetched from the JSON
    subset_imageIds = common_dataframe_utils.extract_values_from_column(
        IN_pandas_dataframe=data_df,
        IN_filter_column_name="train",
        IN_filter_criteria=True,
        OUT_column_name="image_id"
    )
    _download_images_by_id(subset_imageIds, 'train', OUT_PATH_train_images)
    print(f"[INFO] I downloaded {len(subset_imageIds)} images")
    
    # Download all images marked in column "train" as "False" (i.e. validation images) corresponding to the image IDs
    # fetched from the JSON
    subset_imageIds = common_dataframe_utils.extract_values_from_column(
        IN_pandas_dataframe=data_df,
        IN_filter_column_name="train",
        IN_filter_criteria=False,
        OUT_column_name="image_id"
    )
    _download_images_by_id(subset_imageIds, 'train', OUT_PATH_validation_images)
    
    return data_df


# endregion

def train_batch(data, encoder, decoder, optimizer, criterion):
    encoder.train()
    decoder.train()
    images, captions, lengths = data
    images = images.to(device)
    captions = captions.to(device)
    targets = pack_padded_sequence(captions, lengths.cpu(), batch_first=True)[0]
    features = encoder(images)
    outputs = decoder(features, captions, lengths)
    loss = criterion(outputs, targets)
    decoder.zero_grad()
    encoder.zero_grad()
    loss.backward()
    optimizer.step()
    return loss

@torch.no_grad()
def validate_batch(data, encoder, decoder, criterion):
    encoder.eval()
    decoder.eval()
    images, captions, lengths = data
    images = images.to(device)
    captions = captions.to(device)
    targets = pack_padded_sequence(captions, lengths.cpu(), batch_first=True)[0]
    features = encoder(images)
    outputs = decoder(features, captions, lengths)
    loss = criterion(outputs, targets)
    return loss

def load_image(image_path, transform=None):
    image = Image.open(image_path).convert('RGB')
    image = image.resize([224, 224], Image.LANCZOS)
    if transform is not None:
        tfm_image = transform(image)[None]
    return image, tfm_image

@torch.no_grad()
def load_image_and_predict(image_path):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))
    ])

    org_image, tfm_image = load_image(image_path, transform)
    image_tensor = tfm_image.to(device)
    encoder.eval()
    decoder.eval()
    feature = encoder(image_tensor)
    sentence = decoder.predict(feature)[0]
    show(org_image, title=sentence)
    return sentence

# endregion

# ===================
# MAIN
# ===================

print("\n[START] Start of execution {}\n".format(time.strftime("%H:%M:%S")))

# region [SECTION] Generate a dataframe with image information and download them (after checking, if they exist)
print("\n[INFO] Checking, if images were already downloaded.")
if os.path.isfile(path=OUT_image_train_not_train_csv):
    
    # count lines in data.csv with each line representing one image
    line_count = 0
    with open(OUT_image_train_not_train_csv, 'r') as file:
        for line in file:
            line_count += 1

    # comapring the set variable max_download_images with the lines counted in the file data.csv, assuming,
    # that the script previously was executed properly and all images according data.csv were downloaded.
    print(f"[ATTENTION]\tI just discovered, that I already previously downloaded files spending significant efforts!!\n"
          f"\t\tYou now ask me to download {max_download_images} images and I already once downloaded {line_count-1}!")
    
    user_decision = input(f"\n[INPUT] Do you really want to download {max_download_images} images again (y/n): "
                          f"").lower()
    
    if user_decision == "y":
        # delete csv and image files
        try:
            # delete previously downloaded data.csv file
            os.remove(OUT_image_train_not_train_csv)

            # delete all images saved in training_folder
            for filename in os.listdir(OUT_PATH_train_images):
                file_path = os.path.join(OUT_PATH_train_images, filename)
                try:
                    os.remove(file_path)
                except OSError as error:
                    print(f"Error deleting {file_path}: {error}")

            # delete all images saved in validation_folder
            for filename in os.listdir(OUT_PATH_validation_images):
                file_path = os.path.join(OUT_PATH_validation_images, filename)
                try:
                    os.remove(file_path)

                except OSError as error:
                    print(f"Error deleting {file_path}: {error}")
            print(f"[INFO] All files deleted successfully")
        except OSError as error:
            print(error)
            
        # after deleting everything, rebuild the data.csv file containing images to be downloaded and execute the
        # download
        print("[PROCESSING] Datafile is being generated and images will be downloaded.")
        data = download_images(IN_image_json_file=IN_image_json,
                               IN_max_amount_of_images=max_download_images,
                               IN_validation_share= share_of_validation_images
                               )
    else:
        print("[INFO] Great, we use the downloaded images! Good choice!")
        data = pd.read_csv(OUT_image_train_not_train_csv)

else:
    print("[PROCESSING] Datafile is being generated and images will be downloaded.")
    data = download_images(IN_image_json_file=IN_image_json,
                           IN_max_amount_of_images=max_download_images,
                           IN_validation_share = share_of_validation_images
                           )
# endregion

# region [SECTION] Generate a vocabulary of all the unique words present in all captions from the dataframe

# extract a list of captions from the dataframe
all_captions = data[data['train']]['caption'].tolist()
# clean the list
all_tokens = common_tokenizer_utils.get_clean_list_of_words(IN_list_of_items=all_captions)

# Generate a field object to later being used for building a vocabulary
""" [INFO] A Field object is a PyTorch object that is designed to handle a specific type of data, such as text or
numerical values, and for which I can define the methods and rules for how that data should be preprocessed and
represented.
For example in regards to text data, I can create a Field object and specify various parameters, such as the
tokenization method, the vocabulary size, and the padding strategy. You can also define additional methods,
such as custom preprocessing or postprocessing steps, to be applied to the data.
"""

# Parameter:
# sequential = False: This indicates that the field is not sequential, i.e., it does not represent a sequence of words
# or tokens.
# init_token = '': This sets the initial token for the field to an empty string.
# eos_token = '': This sets the end-of-sequence token for the field to an empty string.
captions = torchtext.data.Field(sequential=False, init_token='', eos_token='')

# Finally, the build_vocab method of "captions"  is called with the "all_tokens" list as argument. This builds
# a vocabulary for the field using the tokens in "all_tokens".
# It generates two dictionaries: itos = integer_to_string and stoi = string_to_integer, where each word is mapped.
captions.build_vocab(all_tokens)

# We only need the captions vocabulary components of the captions object, so in the following code, we create a dummy
# "vocab" object to later store the vocabulary.
vocab = Vocab()

# The insert method is being called on the itos list to insert "<pad>" at index 0,
captions.vocab.itos.insert(0, "<pad>")

# copy the itos list from the object "captions" to the empty object "vocab"
vocab.itos = captions.vocab.itos

# This code sets up a defaultdict called vocab.stoi which maps words in the vocabulary (s) to their corresponding
# index in the vocabulary (i).
# 1) Set up a default value for the defaultdict, i.e. any words not found in the vocabulary will default to the index
# of the <unk> = unknown token.
vocab.stoi = defaultdict(lambda: captions.vocab.itos.index("<unk>"))

# Set the index of the <pad> token to 0.
vocab.stoi["<pad>"] = 0

# Iterate through each word s in the vocabulary of captions and sets the corresponding index in
# vocab.stoi to be one greater than the index in captions.vocab.stoi. This is because vocab.stoi has already reserved
# index 0 for <pad>.
for s, i in captions.vocab.stoi.items():
    vocab.stoi[s] = i+1
# endregion

# region [SECTION] Define the training and validation dataset and data loaders:

trn_ds = CaptioningData(root=OUT_PATH_train_images,
                        df=data[data['train']],
                        vocab=vocab)

val_ds = CaptioningData(OUT_PATH_validation_images, data[~data['train']], vocab)

# endregion

image, target, caption = trn_ds.choose()
show(image, title=caption, sz=5); print(target)

encoder = EncoderCNN(256).to(device)

trn_dl = DataLoader(trn_ds, 32, collate_fn=trn_ds.collate_fn)
val_dl = DataLoader(val_ds, 32, collate_fn=val_ds.collate_fn)
inspect(*next(iter(trn_dl)), names='images,targets,lengths')

# print(summary(encoder,torch.zeros(32,3,224,224).to(device)))

encoder = EncoderCNN(256).to(device)
decoder = DecoderRNN(256, 512, len(vocab.itos), 1).to(device)
criterion = nn.CrossEntropyLoss()
params = list(decoder.parameters()) + list(encoder.linear.parameters()) + list(encoder.bn.parameters())
optimizer = torch.optim.AdamW(params, lr=1e-3)
n_epochs = 6
log = Report(n_epochs)

for epoch in range(n_epochs):
    if epoch == 5:
        optimizer = torch.optim.AdamW(params, lr=1e-4)
    N = len(trn_dl)
    for i, data in enumerate(trn_dl):
        trn_loss = train_batch(data, encoder, decoder, optimizer, criterion)
        pos = epoch + (1+i)/N
        log.record(pos=pos, trn_loss=trn_loss, end='\r')

    N = len(val_dl)
    for i, data in enumerate(val_dl):
        val_loss = validate_batch(data, encoder, decoder, criterion)
        pos = epoch + (1+i)/N
        log.record(pos=pos, val_loss=val_loss, end='\r')

    log.report_avgs(epoch+1)

log.plot_epochs(log=True)

files = Glob(OUT_PATH_validation_images)
load_image_and_predict("/Users/swmoeller/python/prj_image_captioning_e2e/src/models/20090718_012_WaterCity-Xitang.jpg")
#for _ in range(5):
#    load_image_and_predict(choose(files))

print("\n[END] End of execution: ", time.strftime("%H:%M:%S"))
