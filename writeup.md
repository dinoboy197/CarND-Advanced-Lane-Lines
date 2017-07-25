**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[undistorted_chessboard]: ./examples/undistorted_chessboard.png "Undistorted"
[undistorted_road]: ./examples/undistorted_road.png "Road Undistorted"
[thresholded_binary]: ./examples/thresholded_binary.png "Binary Example"
[warped_road]: ./examples/warped_road.png "Warp Example"
[polynomial_fit]: ./examples/polynomial_fit.png "Fit Visual"
[polynomial_fit_limited]: ./examples/polynomial_fit_limited.png "Fit Visual"
[final_lane]: ./examples/final_lane.png "Output"

---

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for camera matrix and distorion coefficients computation is in the method [`compute_calibration_mtx_and_distortion_coeff()`](https://github.com/dinoboy197/CarND-Advanced-Lane-Lines/blob/master/find_lanes.py#L24-L70).

This method starts by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image. Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image. `img_points` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.

Next, each chessboard calibration image is looped over. Each image is converted to grayscale, then `cv2.findChessboardCorners` is used to detect the corners. Corners detected are made more accurate by using `cv2.cornerSubPix` with a suitable search termination criteria, then the object points and image points are added for later calibration.

Finally, the image points and object points are used to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` method.

I applied this distortion correction to the test image using `cv2.undistort()` in the [`correct_distortion()`](https://github.com/dinoboy197/CarND-Advanced-Lane-Lines/blob/master/find_lanes.py#L73-L75) method and obtained this result:

![alt text][undistorted_chessboard]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

The distortion correction method `correct_distortion()` is used on a road image, as can be seen in this before and after image:
![alt text][undistorted_road]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

To create a thresholded binary image, I created a method called [`create_thresholded_binary()`](https://github.com/dinoboy197/CarND-Advanced-Lane-Lines/blob/master/find_lanes.py#L78-L113). This method detects horizontal line segments through a Sobel x gradient computation, white lines through a identifying high signal in the L channel of the LUV color space, and yellow lines through identifying low (yellow) signal in the B channel of the LAB color space. Any pixel identified by any of the three filters contributes to the binary image.

Here is an example of an original image and a thresholded binary created from it:

![alt text][thresholded_binary]

Note that the thresholding detection picks up many other pixels that are not part of the yellow or white lane lines, though the selected pixel density in the lanes are significantly greater than the overall noise in the thresholded binary image so as to not confuse the lane line detection in a future step.

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

I compute a perspective transform in the method [`compute_perspective_transform_matrices()`](https://github.com/dinoboy197/CarND-Advanced-Lane-Lines/blob/master/find_lanes.py#L116-L123). This method uses a hardcoded trapezoid and rectangle determined by observation in the original unwarped image.

This resulted in the following source and destination points:

| Source        | Destination   |
|:-------------:|:-------------:|
| 589, 455      | 300, 0        |
| 692, 455      | 1030, 0       |
| 1039, 676     | 980, 719      |
| 268, 676      | 250, 719      |

I verified that my perspective transform was working as expected by viewing the pre and post-transformed images:

![alt text][warped_road]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

I used two methods of identifying lane lines in a thresholded binary image and fitting with a polynomial. The first method identifies pixels by a naive sliding window detection algorithm; the second method identifies pixels by starting with a previous line fit as a starting point. Both methods use common code in [`fit_lane_line_polynomials()`](https://github.com/dinoboy197/CarND-Advanced-Lane-Lines/blob/master/find_lanes.py#L221-L272) to pick the method to use, and fall back to naive sliding window search if the previous line fit does not perform.

In the first method, [`fit_lane_line_polynomial()`](https://github.com/dinoboy197/CarND-Advanced-Lane-Lines/blob/master/find_lanes.py#L126-L180), the thresholded binary image is scanned on nine individual horizontal slices of the image. Slices start at the bottom and move up, selecting from the nearest to farthest point on the road. In each slice, a box starts at the horizontal location with the most highlighted pixels, and moves to the left or right at each step "up" the image based where most of the highlighted pixels in the box are detected, with some constraints on how far to the right the image can move and how big the windows are. Any pixels caught in each sliding window are used for a 2nd degree polynomial curve fit. This method is performed twice for each image, to attempt to capture both left and right lanes.

Here is an example of a thresholded binary with sliding windows and polynomial fit lines drawn over:

![alt text][polynomial_fit]

In the second method, [`fit_lane_line_polynomial_with_previous_fit()`](https://github.com/dinoboy197/CarND-Advanced-Lane-Lines/blob/master/find_lanes.py#L183-L218), two previous polynomial fit lines are used (likely taken from a previous frame of video) to generate a "channel" around the line with a given margin. Only highlighted pixels in the "channel" around the line are used for the next fit line. This method can ignore more noise that the first method would be subject to; this can come in particularly useful in areas of shadow or many yellow or white areas in the image that are not lane lines. This method can also fail if no pixels are detected in the "channel" around the previous line.

Here is an example of a thresholded binary with previous fit channels and polynomial fit lines drawn over:

![alt text][polynomial_fit_limited]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

Radius of curvature computation my program is intertwined with curve and lane line detection smoothing, which occurs in the methods [`determine_curve_radius_and_lane_points()`](https://github.com/dinoboy197/CarND-Advanced-Lane-Lines/blob/master/find_lanes.py#L287-L347) and [`radius_of_curvature()`](https://github.com/dinoboy197/CarND-Advanced-Lane-Lines/blob/master/find_lanes.py#L350-L357).

In the first method, the radius of curvature is determined by computing [the radius of curvature equation](http://www.intmath.com/applications-differentiation/8-radius-curvature.php) (straightforward algebra).

In the second method (which provides a small degree of curvature and lane smoothing from video frame to frame), the raw lane lines detected in the previous step are combined with the lane lines found in the previous ten frames of video. Lane lines whose curvatures are more than 1.5 standard deviations from the median are ignored, and the remaining curvatures are averaged. The lane lines with the curvature closest to the average are selected for both drawing onto the final image, as well as for the chosen curvature.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

After the lane line is chosen by the smoothing algorithm above, the lane line pixels are drawn back onto the image, resulting in this:

![alt text][final_lane]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

My lane detection algorithm was run on three videos:
* [Project video](project_video_processed.mp4) - lane finding is quite robust, having some slight wobbles when the vehicle bounces across road surface changes and when shadows appear in the roadway
* [Challenge video](challenge_video_processed) - lane finding is useful throughout the entire video, though the lane detection algorithm selects a shadow edge rather than the yellow lane line for a portion of the video
* [Harder challenge video](harder_challenge_video_processed) - lane finding is primitive, staying with the lane for only a small portion of the time.

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

 
