# mnist

The MNIST database contains binary images of handwritten digits commonly used to train image processing systems. The digits were collected from among Census Bureau employees and high school students. The database contains 60,000 training digits and 10,000 testing digits, all of which have been size-normalized and centered in a fixed-size image of 28 × 28 pixels. Many methods have been tested with this dataset and in this project, you will get a chance to experiment with the task of classifying these images into the correct digit using some of the methods you have learned so far.

## What's inside

- part1/linear_regression.py where you will implement linear regression
- part1/svm.py where you will implement support vector machine
- part1/softmax.py where you will implement multinomial regression
- part1/features.py where you will implement principal component analysis (PCA) dimensionality reduction
- part1/kernel.py where you will implement polynomial and Gaussian RBF kernels
- part1/main.py where you will use the code you write for this part of the project

To get warmed up to the MNIST data set run python main.py. This file provides code that reads the data from mnist.pkl.gz by calling the function get_MNIST_data that is provided for you in utils.py. The call to get_MNIST_data returns Numpy arrays:

1. train_x : A matrix of the training data. Each row of train_x contains the features of one image, which are simply the raw pixel values flattened out into a vector of length . The pixel values are float values between 0 and 1 (0 stands for black, 1 for white, and various shades of gray in-between).
2. train_y : The labels for each training datapoint, also known as the digit shown in the corresponding image (a number between 0-9).
3. test_x : A matrix of the test data, formatted like train_x.
4. test_y : The labels for the test data, which should only be used to evaluate the accuracy of different classifiers in your report.
   Next, we call the function plot_images to display the first 20 images of the training set. Look at these images and get a feel for the data (don't include these in your write-up).
