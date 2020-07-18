# Directional-object-tracking-with-TFLite-and-Edge-TPU

This code performs object detection and tracking using a pre-trained Tensor Flow Lite (TFLite) model.
In addition it can track for each unique object how it is moving through the frame of vision.

As with my previous project there is a filter built into the code so the type of object to track can be specified. For example it can be set to only track all 'people' moving through the frame but ignore other objects such as dogs or bikes.

Practical use: Automated throughfare or doorway monitoring to under stand flow of 'people' in each direction or calculate how many 'people' are 'inside' at any one time.

# Credits:
Main boilerplate for object detection and labelling using TFLite & Edge TPU - Evan Juras
https://github.com/EdjeElectronics/TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi
This code was based off the TensorFlow Lite image classification example at:
https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/examples/python/label_image.py
Centroid tracking helper code for object tracking:
https://github.com/lev1khachatryan/Centroid-Based_Object_Tracking 

