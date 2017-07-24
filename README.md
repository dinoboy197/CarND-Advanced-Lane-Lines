## Advanced Lane Finding
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

Overview
---
In this project, your goal is to write a software pipeline to identify the lane boundaries in a video.

Submission Contents
---
* [README.md](README.md) - this readme
* [writeup.md](writeup.md) - an explanation of the solution
* [find_lanes.py](find_lanes.py) - script used to identify lane boundaries
* [camera_cal](camera_cal) - calibration images for the camera image distortion correction
* [test_images](test_images) - individual still frames on which to test the lane boundary detection script
* [output_images](output_images) - individual still frames from [test_images](test_images) which have lane boundaries drawn on them
* [project_video.mp4](project_video.mp4), [challenge_video.mp4](challenge_video.mp4), [harder_challenge_video.mp4](harder_challenge_video.mp4) - three videos to be used as input for lane boundary detection (in increasing order of difficulty)
* [project_video_processed.mp4](project_video_processed.mp4), [challenge_video_processed.mp4](challenge_video_processed.mp4), [harder_challenge_video_processed.mp4](harder_challenge_video_processed.mp4) - three videos above which have lane boundaries drawn on them

The Project
---

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

The images for camera calibration are stored in the folder called `camera_cal`.  The images in `test_images` are for testing your pipeline on single frames.  If you want to extract more test images from the videos, you can simply use an image writing method like `cv2.imwrite()`, i.e., you can read the video in frame by frame as usual, and for frames you want to save for later you can write to an image file.

To help the reviewer examine your work, please save examples of the output from each stage of your pipeline in the folder called `ouput_images`, and include a description in your writeup for the project of what each image shows.    The video called `project_video.mp4` is the video your pipeline should work well on.

The `challenge_video.mp4` video is an extra (and optional) challenge for you if you want to test your pipeline under somewhat trickier conditions.  The `harder_challenge.mp4` video is another optional challenge and is brutal!

If you're feeling ambitious (again, totally optional though), don't stop there!  We encourage you to go out and take video of your own, calibrate your camera and show us how you would implement this project from scratch!
