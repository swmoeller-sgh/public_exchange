
Concepts
==============

Overview
-----------
- Using a CNN, recognize the different features of an image.
- Cut off the last layer of the CNN (i.e. classification layer).
- Feed the recognized features from the CNN into a NLP (e.g. BERT or others).
- Train the combined NN (CNN and NLP) using flickr dataset containing images and their captions
- read a batch of images and generate a dictionary containing image name and its caption


CNN neural network(s)
-----------------
best-in-class: Faster R-CNN 
Region-based Convolutional Neural Networks. This method combines region proposals for object segmentation and high 
capacity CNNs for object detection 
reference: https://journalofbigdata.springeropen.com/articles/10.1186/s40537-021-00434-w 



- load an image nn and its pre-trained weights

Dataloader
---------


Training
---------

Inference
---------


Reference
===============
- Tutorial on using Faster R-CNN: https://towardsdatascience.com/understanding-and-implementing-faster-r-cnn-a-step-by-step-guide-11acfff216b0
- 

"""