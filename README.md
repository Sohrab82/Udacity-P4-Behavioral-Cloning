# Behavioral Cloning Project

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

### The Project
---

#### Goals
* Use the simulator to collect data of good driving behavior 
* Design, train and validate a model that predicts a steering angle from image data
* Use the model to drive the vehicle autonomously around the first track in the simulator. 

#### Files
* [model.py](./model.py) (script used to create and train the model)
* [drive.py](./drive.py) (script to drive the car)
* [model.h5](https://drive.google.com/file/d/1btzSviMPpu1xjrLYtc1MkZ2lOISEPh9T/view?usp=sharing) (a trained Keras model)
* [video.mp4](./video.mp4) (a video recording of your vehicle driving autonomously around the track)


#### Dependencies

* [CarND Term1 Starter Kit](https://github.com/udacity/CarND-Term1-Starter-Kit)


### Model
* For the neural network model, transfer learning is used by adding a input and output layers to the Keras' InceptionV3 model.
* Adam optimizer with default 1e-3 learning rate is used.
* `generator` function is used to read the images in batches and provide them to the fit function as requested. Loading all the input data into variables deemed impossible due to large size of the input data.
* Input data is split into training and validation data (80% vs 20%)
* The car managed to drive in both directions on the track with the following model.

#### Preprocessing
* A moving average filter with a with of 3 samples is applied to the steering wheel angles as the data obtained with the keyboard does not reflect the true steering angle at consequitive frames.
* `flip_lr` is applied to every image to balance the number of right and left turns. In the original map, the number of left turns is much higher.


#### Input layers
* Input layer with input size of (160, 320, 3), the size of the images aquired from the simulator
* Lambda layer to normalize the input images by dividing them by 255.
* Cropping2D layer to crop the top (sky) and bottom (bonet) of the image

#### InceptionV3 model
Inception model has over 22 million trainable parameters. In this project, pretrained model is loaded an the parameters are to `trainable=False`. Te top layers (the dense layers at the end) has been removed to be able to add our own layers and use the model for regression (the steering angle).
At the output of the top (last) layer, 2048 channels exist which for our current input size, will be 1x8 (i.e (None, 1, 8, 2048).

#### Output layers
* [SeparableConv2D](https://towardsdatascience.com/a-basic-introduction-to-separable-convolutions-b99ec3102728) layer with kernel size of 1x4 to apply convolution on the 2048 channels from the Inception model. SeparableConv2D has been used instead of Conv2D to remove the number of trainable parameters. Test results showe lower validation loss for this type of convolution.
* Flatten layer
* Dense layer with 64 units and relu
* Dense layer with 1 output. PLEASE NOTE: you don't need relu for this layer as the steering angle values can be both positive and negative.

