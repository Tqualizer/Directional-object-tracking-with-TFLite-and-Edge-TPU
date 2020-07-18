# Directional object tracking with TFLite and optional Edge TPU boost
This code performs object detection and tracking using a pre-trained Tensor Flow Lite (TFLite) model. In addition it can track for each unique object how it is moving through the frame of vision i.e. if it is moving up /  down / left / right or just stationary

Credits:
* Main boilerplate for object detection and labelling using TFLite & Edge TPU - Evan Juras https://github.com/EdjeElectronics/TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi
* This code was based off the TensorFlow Lite image classification example at: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/examples/python/label_image.py
* Centroid tracking helper code for object tracking: https://github.com/lev1khachatryan/Centroid-Based_Object_Tracking 

As with my previous project there is a filter built into the code so the type of object to track can be specified. For example it can be set to only track all 'people' moving through the frame but ignore other objects such as dogs or bikes. The results are most accurate when the camera is aimed at a fixed area in which objects will move through over the course of a few seconds. 

Practical uses: By counting the number of objects passing thought the frame in the each direction the code could easily be set as an automated throughfare or doorway monitoring tool under stand flow of 'people' in each direction or calculate how many 'people' are 'inside' at any one time. Add a log file of the movement timestamps to perform further time series analysis of movement e.g. busiest times.

<img src="https://github.com/Tqualizer/Directional-object-tracking-with-TFLite-and-Edge-TPU/blob/master/2020-07-18-174302_1920x1080_scrot.png" width ="700" />



## Introduction
This is my second Computer Vision project. I built this using a Raspberry Pi 3B+, standard IR camera and Coral Edge TPU accelerator. 
I had seen some videos and guides on object tracking and thought it would be interesting to try and infer more about how the objects were using. This could also have useful commercial uses for businesses looking for an automated way of tracking number of people inside or traffic drones looking at flows and bust times.

<img src="https://github.com/Tqualizer/Directional-object-tracking-with-TFLite-and-Edge-TPU/blob/master/IMG_20200718_184111.jpg" width ="700" />

## The main steps are as follows:
1. 




1. (optional) **Customisation**
 * Select a custom model and number of objects (as described in the repo referenced in step 1). 
 
 For this example I used the same coco model as the boilerplate code but depending on what you want to detect and how accurate you need the model to be, other models can be easily referenced in the code instead. Check out https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md for more resources or have a go at training your own model if you have the necessary hardware https://github.com/EdjeElectronics/TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10.
 
## How it works

## Appendix: Remote logging (Windows 10 example) See my previous guide



