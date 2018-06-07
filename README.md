# Automatic-Image-Colorization

Our task is to convert given grey-scale image into colorized(RGB) image.

## Introduction

Given a grayscale image input, our objective is to find out its believable color image without any manual intervention. Accomplishing this task is a difficult problem because there is often no “correct” attainable color for a given image as the color of the t-shirt can be anything from white to red or black.  

## Lab Color Space

It is also known as [CIELAB color space](https://en.wikipedia.org/wiki/CIELAB_color_space). 

It is expressed in three numerical values *(L,a,b)* as
- **L(Luminance)** for the lightness 
- **a** for the green–red color components
- **b** for the blue–yellow color components

![image](images/lab_space.jpg)

Each and every image while training phase is first converted to the Lab color space then passed it further for processing. The Luminance(L) channel is nothing but the grayscale image of the color image given for the training.

>*Why do we use Lab color space?*

> The reason for using this over RGB because euclidean distance in Lab is more similar to how humans perceive color differences[1].

## Approaches

Here to build a solution, we are exploring various options from traditional hand-picked features to modern deep learning techniques.

### 1. Feed Forward Neural Network

As the *Luminance*(**L**) channel is the grayscale image we have to find out channel **a** and **b**. 

#### i) Regression Problem

Here we treat the problem of predicting ab from L as a *Regression problem*, we train two Separate MLP with (500,400,300) number of Units in the 3 hidden layer and try to predict **a** and **b**. 
Here we first resize the image to (200,200) size and then feed each pixel value into the Neural Net and solve the problem like a regression.

![image](images/NN.png)

> Simple Feed Forward Neural Network takes pixel as input gives a,b value as Output.

**Observation:** The result we get is nowhere near great, but still, we can feel some sort of colorization. It is because only the pixel’s intensity is not enough to predict the color as *Lab color space is a decorrelated colorspace there is no correlation between the L and ab channels*.

#### ii) Classification Problem

First, we discretize the color space to 8x8, by dividing every pixel value with 32 and rounding off to the nearest integer.
Here as a feature, we take the L values of a patch of size 3x3 around the pixel, our feature size becomes 9 values and we try to classify each pixel in one of 64 ab value.

![Output Image](images/NN_1.png)

**Observation:** It is not enough to only pass the raw pixels of the 3x3 patch around the particular pixel. One reason why this is failing can be due to the flattening of the 3x3 feature matrix we are doing to feed it to the neural network, that way it *loses all the spatial information*.

### 2. Combination of Feed Forward Neural Network and Hand Picked Features

Limitation of Data and computation resource we look into hand-picked features, we followed most of the approach from this paper[1]. Here we used the CVCL MIT Opencountry Dataset [link](http://cvcl.mit.edu/database.htm). We took a subset of images (11 images) used that to do the training and we used another subset of 10 images to colorize.

**Flowchart:**

![Image](images/flowchart.png)

> **Note-** Here we use Neural Network instead of SVM. Also, we didn’t use PCA(as shown in the paper), instead of reducing the dimension we directly feed the features to the Neural Network.

### 3. Convolutional Neural Network (CNN)


## References

1. [Colorful Image Colorization](https://arxiv.org/pdf/1603.08511.pdf) by Richard Zhang, Phillip Isola, and Alexei A. Efros
2. [Automatic Colorization of Grayscale Images](http://cs229.stanford.edu/proj2013/KabirzadehSousaBlaes-AutomaticColorizationOfGrayscaleImages.pdf)
