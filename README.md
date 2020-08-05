# Directional object tracking with TFLite and optional Edge TPU boost
This code performs object detection and tracking using a pre-trained Tensor Flow Lite (TFLite) model. In addition, it can track each unique object in terms of how it is moving through the frame of vision i.e. if it is moving up /  down / left / right or just stationary. There are two main scripts: one takes a live video feed from Raspberry Pi camera; the other analyses an existing video file.

<div align="center">
      <a href="https://youtu.be/lGIXOZn9Cvw/">
         <img src="http://img.youtube.com/vi/lGIXOZn9Cvw/0.jpg" style="width:200%;">
      </a>
</div>

Credits:
* Main boilerplate for object detection and labelling using TFLite & Edge TPU - Evan Juras 
https://github.com/EdjeElectronics/TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi
* Centroid tracking helper code for object tracking: 
https://github.com/lev1khachatryan/Centroid-Based_Object_Tracking 

As with my previous project there is a filter built into the code so the type of object to track can be specified. For example it can be set to only track all 'people' moving through the frame but ignore other objects such as dogs or bikes. The results are most accurate when the camera is aimed at a fixed area in which objects will move through over the course of a few seconds. 

**Practical uses:** By counting the number of objects passing thought the frame in the each direction the code could easily be set as an automated throughfare or doorway monitoring tool to understand the flow of 'people' in each direction or calculate how many 'people' are 'inside' at any one time. Add a log file of the movement timestamps to perform further time series analysis of movement e.g. busiest times.

From a personal data perspective it does not store any pictures of the objects or record their features. The code works in realtime from the video stream and only stores co-ordinates.

<img src="https://github.com/Tqualizer/Directional-object-tracking-with-TFLite-and-Edge-TPU/blob/master/Examples/IMG_20200718_184111.jpg" width ="700" />


## Introduction
This is my second Computer Vision project. The main thing holding me back from doing this sooner was that when running Tensor Flow on my Raspberry Pi 3B+ for my previous project, the best framerate I could get was around 3FPS. This worked for object detection but I needed a much higher framerate to continuously track moving objects. I bought a Coral Edge TPU accelerator to help increase the TF calculation speed. This approach meant I could boost the whole process to around 14FPS on a TF Lite model. This was enought to get to work on directional object tracking. 

I built this using a Raspberry Pi 3B+, standard IR camera and Coral Edge TPU accelerator. 
I had seen some videos and guides on object tracking and thought it would be interesting to try and infer more about how the objects were moving. This could also have useful commercial uses for businesses looking for an automated way of tracking number of people inside or traffic drones looking at flows and busy times.

<img src="https://github.com/Tqualizer/Directional-object-tracking-with-TFLite-and-Edge-TPU/blob/master/Examples/InkedCroppedCodeViewer_LI.jpg" width ="700" />

## How to set up the live object direction tracker
1. Follow Evan's guide to getting TensorFlow Lite up and running on the Raspberry Pi: https://github.com/EdjeElectronics/TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi/blob/master/Raspberry_Pi_Guide.md 
1. Save the _TFLite_DirectionLive.py_ and _centroidtracker.py_ files from this repo in the same repo as your TFLite installation.
1. (optional) **Customisation**
 * Select a custom model (as described in the repo referenced in step 1). 
 For this example I used the same coco model as the boilerplate code but depending on what you want to detect and how accurate you need the model to be, other models can be easily referenced in the code instead. Check out https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md for more resources or have a go at training your own model if you have the necessary hardware https://github.com/EdjeElectronics/TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10.

* Define any objects of interest (these must exist in your TFLite model already). You can use the _labelmap.txt_ to check you have the right name.
```
 object_name = labels[int(classes[i])] # Look up object name from "labels" array using class index
 if object_name == 'person':
```

* Set the calculation interval, movement sensitivity and which events to log. 
The **calculation interval** is how many frames between each time centroid distance is measured. The current value of 30 is calibrated for walking speed. Use lower for faster moving objects. 
The **movement sensitivity** setting is important to eliminate false positives from minor frame jitters. Increase the sensitivity for faster moving objects.

```
#see what the difference in centroids is after every x frames to determine direction of movement and tally up total number of objects that travelled left or right
    if obsFrames % 30 == 0:  # Calculation interval _x_ of frames between each time it calculates the distance travelled.
        d = {}
        for k,v in x.items():
            if v[0] > 3:  # This number is the sensitivity for positive x axis movement
                d[k] =  "Left"
                leftcount = leftcount + 1 
            elif v[0]< -3: # This number is the sensitivity for negative x axis movement
                d[k] =  "Right"
                rightcount = rightcount + 1 
            elif v[1]> 3: # This number is the sensitivity for positive y axis movement
                d[k] =  "Up"
            elif v[1]< -3: # This number is the sensitivity for negative y axis movement
                d[k] =  "Down"
            else: 
                d[k] = "Stationary"
        if bool(d):
            print(d, time.ctime()) # prints the direction of travel (if any) and timestamp
```
For my project the order of the IF statements did not matter given that left and right are mutually exclusive. As well as changing the order, you could also account for combinations of x and y axis movements by assigning them to compass directions. For example:

```
            if v[0] > 3 & v[1] >3:
                d[k] =  "North West"
```

4.  **Run** the *TFLite_DirectionLive.py* from your _TFLite_ directory. To safely stop the process and save outputs press 'q' on the object viewer or Ctrl + C in the command line to exit. 
  **Enable Edge TPU** (optional) by adding the arg _--edgetpu_ in the command line. There are also other commands if you want to lower the resolution for example to increase the framerate. 

<img src="https://github.com/Tqualizer/Directional-object-tracking-with-TFLite-and-Edge-TPU/blob/master/Examples/InputSnippet.png" width ="700" />

## How to set up directional tracking for an existing video file
1. Continuing from the steps above, download *TFLite_DirectionTracker.py* from this repo. 

1. Apply any steps above needed for customisation such as specifying the object type you want to track or finetuneing the movement thresholds or sensitivity.

1. When running group_detection_recorded.py from the command line, specify in the input filename in the --args:
```
/tflite1 python3 group_detection_recorded.py --edgetpu --video yourvideofile.mp4
```
The script will then run through your video file labeling the direction of each moving object which meets the criteria you originally specified.

<img src="https://github.com/Tqualizer/Directional-object-tracking-with-TFLite-and-Edge-TPU/blob/master/Examples/output_Moment.jpg" width ="700" />



## How it works

Starting with the boilerplate object detector from 
https://github.com/EdjeElectronics/TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi these were the _main_ steps
1. Integrate the centroid tracking to _centroidTracking.py_ from https://github.com/lev1khachatryan/Centroid-Based_Object_Tracking by adding these snippets below:
```
# initialize our centroid tracker and frame dimensions
ct = CentroidTracker()
objects ={}
old_objects={}

#rects variable
rects =[]

#update the centroid for the objects
objects = ct.update(rects)

# loop over the tracked objects for (objectID, centroid) in objects.items():
# draw both the ID of the object and the centroid of the
# object on the output frame
   text = "ID {}".format(objectID)
   cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
   cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)         
```

2. Expand the object detection code to compare each frame to the previous frame and calculate object movements:

```
# On the next loop set the value of these objects as old for comparison
old_objects.update(objects)

# compare the co-ordinates for dictionaries of interest
def DictDiff(dict1, dict2):
   dict3 = {**dict1}
   for key, value in dict3.items():
       if key in dict1 and key in dict2:
               dict3[key] = np.subtract(dict2[key], dict1[key])
   return dict3

# calculate the difference between this and the previous frame
x = DictDiff(objects,old_objects)

```
3. Insert the actual movement calculations for each direction.
* Get the co-ordinates of where each object is and where to put the label.
```
    difflist= pd.DataFrame.from_dict(x).transpose()
    difflist.columns = ['a','b']
    difflist['index'] = difflist.index
    z = difflist.merge(objectslist,left_on = 'index', right_on = 'index', suffixes=('_diff','_current'))

    dirx = z['c']
    diry = z['d']

    for i,j,k in zip(dirlabels,dirx,diry):
        direction = format(dirlabels[i])
        cv2.putText(frame, direction, (j+ 10, k + 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 60, 255), 2)       
```
* Calculate the direction and set how often you want to refresh the direction.
```
    #see what the difference in centroids is after every x frames to determine direction of movement
    #and tally up total number of objects that travelled left or right
    if obsFrames % 5 == 0: #set this to a higher number for more accurate tallying
      for index,row in z.iterrows():
            
            if row['b'] < -2:
                dirlabels[index] = "Down"
            if row['b'] > 2 :
                dirlabels[index] = "Up"   
            if row['a'] > 2:
                dirlabels[index] = "Left"
            if row['a'] < -2:
                dirlabels[index] = "Right"
            if row['b'] > 3 & row['a'] > 1:
                dirlabels[index] = "Up Left"
            if row['b'] > 3 & row['a'] < -1:
                dirlabels[index] = "Up Right"
            if row['b'] < -3 & row['a'] > 1:
                dirlabels[index] = "Down Left"
            if row['b'] < -3 & row['a'] < -1:
                dirlabels[index] = "Down Right"
            if row['b'] > 30 | row['a'] > 30:
                dirlabels[index] = "" # to ignore direction on the first frame obejects are loaded in

```
### Appendix
For Remote logging or object counting guides see my previous project: https://github.com/Tqualizer/opencv-group-detection/blob/master/README.md



