# Marinminds Realtime image segmentation

This repository contains the code and a trained model to perform realtime segmentation on a video source such as a webcam.
 
## Cloning
Before cloning the project make sure that you have installed git lfs in order to clone the models. More info can be found here: https://git-lfs.github.com/
 
## Requirements
Make sure you have installed the following requirements:

CUDA 10.0  
cuDNN 7.6  
Anaconda 

## Installing Packages
`$BASE_FOLDER` is the location where the code is cloned.  
A conda environment is used to install all the packages in.  

1. Create a new conda environment: `conda create -n marinminds python=3.6`  
2. And activate it: `conda activate marinminds`  
3. Go to the folder where the code is cloned: cd `$BASE_FOLDER`  
4. Install all the required packages: `pip install -r requirements.txt`

## Running the Camera server
To start the camera server all you have to do is: `cd $BASE_FOLDER/segmentation_streamer` and `python server.py`
Optionally you can specify a video file to process with the `--file` flag, the default value is `''`

## Configuration
### Framerate
The framerate of the video file or stream can be set in line 38 of segmentation_streamer/server.py 
### Stream address
The address of the stream to open can be set in line 19 of segmentation_streamer/segm_streaming.py

## Utils
See the utils folder for more information.