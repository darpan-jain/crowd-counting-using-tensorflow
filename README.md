# CROWD COUNTING

## Contents

- [What do we have here?](#introduction)
- [Why pretrained models don't work?](#using-pre-trained-models)
- [What now?](#training-a-custom-model)
- [Does it work?](#results)
- [What all do I need?](#prerequisites)
- [Let's do this!](#usage)

***

### Introduction

This repository contains the code of performing the task of implementing a people counter from an overhead video surveillance camera by using Transfer Learning.
***

### Using pre-trained models

- Tensorflow’s Object Detection API provides pre-trained models for object detection, which are capable of detecting around 90 classes (objects) with `person` being one of the classes.

- On giving test images to a pretrained model, the inference results were not as per requirements. On some instances the model detected the entire image as a person and also missing out on some fairly obvious ones.

<p align="center">
  <img src="https://github.com/darpan-jain/crowd-counter/blob/master/pretrained-results/result1.png" width="640px" height="480px"/></p>

- Clearly, using the pre-trained models is not the way to go. So, ~~Pre-Trained Models~~ :confused:
***

### Training a custom model

A custom model had to trained for accurate implementation. The following steps were taken for the same:

1. **Annotated training data** had to be prepared before being given to the model fot training.
1. ***LabelImg*** was used for this purpose, to draw bounding boxes around objects of interest in the training images.
1. LabelImg gives output as **xml** files containing coordinates of each of the bounding boxes in an image and the associated label of the object.
1. All the xml files were converted to a ***train.csv*** and then into a ***train.record*** format. TFRecord format is required by Tensorflow to perform training of a custom model.
1. Similarly a ***val.record*** was created for validation data.
1. The architecture of the model is based on the **Faster RCNN** algorithm, which is an efficient and popular object detection algorithm which uses deep convolutional networks.
1. The config file of the model was modified. The last 90 neuron classification layer of the network was removed and replaced with a new layer that gives output for only one class i.e. person.
1. The **config file** for the same can be found in `./data/utils/faster_rcnn.config`
1. After training the model, the checkpoint model is saved as `model.pb` file.
1. This model can now be deployed and used for obtaining inferences on crowd images.

- The model can be found on this drive link: ​[Custom Model](https://drive.google.com/open?id=1IBgEyaASf10KUFTCbky9mtruUpyoqDWR)

- Download and place the model in `./data/utils` before executing main.py.
***

### Results
Upon running `main.py`, the results are as shown below. (Refer `./results`)

<p align="center">
  <img src="https://github.com/darpan-jain/crowd-counter/blob/master/results/result0003.jpg" width="640px" height="480px"/></p>
  
<p align="center">
  <img src="https://github.com/darpan-jain/crowd-counting-using-tensorflow/blob/master/results/result0007.jpg" width="640px" height="480px"/></p>

***Note:*** Since the model was trained on only **30** annotated images, the accuracy can be significantly increased by using a larger dataset to build the model.
***


### Prerequisites
All the required dependencies can be install by running the command `pip install -r requirements.txt`
***

### Usage

- Place the images (.jpg) you need to run inference on in `./data/images/test`
- Run `main.py`
- Results will be saved in `./results`

***
