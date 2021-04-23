# Copyright (c) 2017, NVIDIA CORPORATION.  All rights reserved.

@page dwx_object_tracker_drivenetNcameras_sample DriveNetNCameras Sample

The DriveNetNCameras sample is a sophisticated, multi-class, higher
resolution example that uses the NVIDIA proprietary deep neural
network (DNN) to perform object detection on multiple cameras. For the single-camera example,
see @ref dwx_object_tracker_drivenet_sample. For a simpler, basic,
single-class example, see [Object Detector using dwDetector](@ref dwx_object_dwdetector).

The DriveNetNCameras sample shows a more complete and sophisticated implementation of
object detection with multiple cameras built around NVIDIA's proprietary network architecture. It provides
significantly enhanced detection quality when compared to @ref dwx_object_dwdetector. For more information on DriveNetNCameras and how
to customize it for your applications, speak to an NVIDIA solution architect.

![Multiclass object detector on 4 RCCB streams using DriveNet](sample_drivenet4cameras.png)


In the DriveNetNCameras sample, up to 12 RAW streams from video/camera are used as input and DriveNet inference is run on each frame of each camera to
detect objects. The output of DriveNet inference is then clustered and tracked with parameters
defined in the sample.

This sample can be run in 2 modes on NVIDIA<sup>&reg;</sup> DRIVE<sup>&trade;</sup> platforms: camera and video.
On Linux, only video mode is supported.

The DriveNetNCameras sample expects RAW videos or live camera input data from an AR0231 (revision >= 4) sensor with an RCCB color filter
and a resolution of 1920 x 1208, which is then cropped and scaled down by half to 960 x 540 in typical
usage.


## Limitations ##

@note The version of DriveNet included in this release is optimized for daytime, clear-weather data. It
does not perform well in dark or rainy conditions.

The DriveNet network is trained to support any of the following six camera configurations:
* Front camera location with a 60&deg; field of view
* Rear camera location with a 60&deg; field of view
* Front-left camera location with a 120&deg; field of view
* Front-right camera location with a 120&deg; field of view
* Rear-left camera location with a 120&deg; field of view
* Rear-right camera location with a 120&deg; field of view

The DriveNet network works on any of those camera positions without additional configuration changes.

The network is trained primarily on data collected in the United States. It may have reduced
accuracy in other locales, particularly for road sign shapes that do not exist in the U.S.

@note The sample does not use foveal detection, hence the detection quality might be worse then in a normal
drivenet sample. However, the sample demonstrates how to feed multiple images at once to the DriveNet. Since
DriveNet supports a batch of 2, we are doubling the throughput of the images processed by the network, by
taking a hit in the detection quality. In order to achieve best detection quality, refer to foveal detection
mode in sample_drivenet.

#### Running the Sample

The command line for running the sample on 3 videos on DRIVE platforms:

    ./sample_drivenetNcameras --input-type=video --videos=<video_file_1.raw>,<video_file_2.raw>,<video_file_3.raw>

The command line for running the sample on multiple cameras on DRIVE platforms:

    ./sample_drivenetNcameras --input-type=camera --camera-type=<rccb camera type> --selector-mask=<mask for cameras>

where `<rccb camera type>` is one of the following: `ar0231-rccb`, `ar0231-rccb-ssc`, `ar0231-rccb-bae`, `ar0231-rccb-ss3322`, `ar0231-rccb-ss3323`

The command line for running the sample on Linux with 3 videos:

    ./sample_drivenetNcameras --videos=<video_file_1.raw>,<video_file_2.raw>,<video_file_3.raw> --stopFrame=<frame_idx> --skipFrame=<skip_frame_factor> --skipInference=<skip_inference_factor>

This runs `sample_drivenetNcameras` until frame `<frame_idx>`. Default value is 0, for which the sample runs endlessly.
`skipFrame` can be used to skip processing frame, e.g., a value of 2 will make the sample process 1 every 3 frames.
Default value is 0, for which the sample process all frames. `skipInference` can be used to skip inference,
e.g., a value of 2 will make the sample run inference on 1 frame every 3 frames. Default value is 0, for which the sample runs inference one all frames.
It combines with `skipFrame`, if both are set.

#### Output

The sample creates a window, displays videos in a grid, and overlays bounding boxes for detected objects.
The color of the bounding boxes represent the classes that it detects:

    Red: Cars
    Green: Traffic Signs
    Blue: Bicycles
    Magenta: Trucks
    Orange: Pedestrians
