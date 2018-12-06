
# Self-Driving Car Traffic Light Detection and Classification

## Overview
This repository contains TensorFlow implementation of traffic light detection and classification task for Udacity annotated self-driving dataset.
![](/out/demo.gif)

### Method
In order to detect traffic lights, a pre-trained model is used (Single Shot Multibox Detector (SSD) with Inception v2 from the TensorFlow Zoo on MS-COCO dataset).  The object detector is lightweight. However, the detection accuracy is not high under some conditions such as occlusion. More complex object detectors can perform better.

After the traffic light is detected in an image, a simple method is exploited to classify the color of the traffic light. To classify the color of the traffic light, brightness feature that uses HSV color space is employed. This method cannot achieve very accurate results. However, it is a fast method for real-time inference. For better accuracy, a deep learning based method could be used.

An output video is generated after detection and classification. Find the output video under `out/` directory.

### Dataset
The dataset is taken from https://github.com/udacity/self-driving-car/tree/master/annotations#dataset-2.

## Files
```
model/ - Model files folder
object-dataset/ - Dataset images folder
out/ - Result folder
detectRecognize.py - Loads the model file and detect and recognize traffic light(s) for given input image(s).
```

## Dependencies
Tests are performed with following version of libraries:

+ Python 3.5
+ Numpy 1.15.2
+ TensorFlow 1.5.0
+ OpenCV-Python
+ Pillow

Ubuntu 14.04 LTS is used for the tests.

## Running
+ Download Udacity Annotated Driving Dataset: [http://bit.ly/udacity-annotations-autti](http://bit.ly/udacity-annotations-autti).
+ Extract the file.
+ Download the model file trained on COCO dataset: http://download.tensorflow.org/models/object_detection/ssd_inception_v2_coco_2017_11_17.tar.gz. Extract and locate the files under `model/`.
+ Run the command below for all testing set of Udacity Annotated Driving Dataset:
	```
	$ python detectRecognizeLight.py
	```
+ Result video will be located under `out/` directory.

## License
The source code is licensed under [GNU General Public License v3.0](./LICENSE).
