# Miscellaneous Utilities
Here are some utility scripts to help working with point clouds and neural nets.

## meta_to_pb.py
Create .pb file from .meta file. Path to .meta file can be changed on line 5.

## pb_to_tensorboard.py
Open .pb file visualization in tensorboard

## streaming<span></span>.py
Non-working prototype of alternative to segmentation_streamer/server.py. Attempts to send result frames over rtsp stream instead. Requires openCV to be compiled with GStreamer support. 

## velo_to_bin.py
Convert .csv file from veloview to .bin file for usage with annotation tools and neural nets. Input and output paths can be set on lines 7 and 8.

## velo_to_ply.py
Convert .csv file from veloview to .ply file. Input and output paths can be set on lines 3 and 4.

## visualize_pointcloud.py
Renders visualization of pointcloud from .bin file. Input path can be set on line 4.
