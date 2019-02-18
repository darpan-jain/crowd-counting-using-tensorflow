# CROWD COUNTER

### INTRODUCTION

This repository contains the code of performing the task of implementing a people counter from an overhead video surveillance camera by using Transfer Learning.

### USING PRE-TRAINED MODELS

Tensorflow’s Object Detection API provides pre-trained models for object detection, which are capable of detecting around 90 classes (objects) with ‘person’ being one of the classes.

On giving test images to a pretrained model, the inference results were not as per requirements. On some instances the model detected the entire image as a person and also missing out on some fairly obvious ones.

![PT Result1](https://github.com/darpan-jain/crowd-counter/blob/master/pretrained-results/result1.png)


### TRAINING CUSTOM MODEL

A custom model had to trained for accurate implementation. The following steps were taken for the same.

1. **Annotated training data** had to be prepared before being given to the model.
2. ***LabelImg*** was used for this purpose, to draw bounding boxes around objects of interest in the training images.
3. LabelImg gives output as **xml** files containing coordinates of each of the bounding boxes in an image and the associated label of the object.
4. All the xml files were converted to a ***train.csv*** and then into a ***train.record*** format. TFRecord format is required by Tensorflow to perform training of a custom model.
5. Similarly a ***val.record*** was created for validation data.
6. The architecture of the model is based on the **Faster RCNN** algorithm, which is an efficient and popular object detection algorithm which uses deep convolutional networks.
7. The model config file was modified for the purpose of this assignment.
8. The last 90 neuron classification layer of the network was removed and replaced with a new layer that gives output for only one class i.e. person.
9. The **config file** for the same can be found in `./data/utils/faster_rcnn.config`
10. After training the model, the checkpoint model is saved as `model.pb` file.
11. This model can now be deployed and used for obtaining inferences
12. The test images were the first 10 frames of the mall dataset.

The model can be found on this drive link: ​[Custom Model](https://drive.google.com/open?id=1IBgEyaASf10KUFTCbky9mtruUpyoqDWR)

Please download and place the model in `./data/utils` before executing main.py.

### RESULTS
Upon running `main.py`, the results are saved as shown below. (Refer `./results`)

![My Model result](https://github.com/darpan-jain/crowd-counter/blob/master/results/result0003.jpg)

### PREREQUISITES
All the required dependencies can be install by running the command `pip install -r requirements.txt`
