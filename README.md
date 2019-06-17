# Marinminds Realtime image segmentation

This repository contains the code and a trained model to perform realtime segmentation on a video source such as a webcam.
 
## Requirements
Make sure you have installed the following requirements:

CUDA 10.0  
cuDNN 7.6  
Anaconda 

## Installing Packages
`$BASE_FOLDER` is the location where the code is cloned.  
A conda environment is used to install all the packages in.  

1. Create a new conda environment: `conda create -n marinminds python=3.6`  
2. And activate it: `conda activate marinminds.6`  
3. Go to the folder where the code is cloned: cd `$BASE_FOLDER`  
4. Install all the required packages: `pip install -r requirements.txt`

## Running the Camera server
To start the camera server all you have to do is: `python server.py`
Optionally you can specify another camera_id with the `--camera_id` flag, the default value is 0