# Advanced Lane Finding

from collections import deque
import cv2
import glob
import numpy as np
import os
from moviepy.editor import VideoFileClip
import matplotlib.pyplot as plt

ym_per_pix = 30/720 # meters per pixel in y dimension (given by input dataset authors)
xm_per_pix = 3.7/700 # meters per pixel in x dimension (given by input dataset authors)
previous_left_fit = None
previous_right_fit = None
previous_left_curve_radius = deque([])
previous_right_curve_radius = deque([])
previous_left_fitx = deque([])
previous_right_fitx = deque([])
debug_image = False

def compute_calibration_mtx_and_distortion_coeff():
    # additional corner finding criteria for closer corner detection
    corner_finding_termination_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    obj_points = []
    img_points = []

    # chessboard calibration images have 9 horizontal and 6 vertical inside corners
    corners_x = 9
    corners_y = 6
    objp = np.zeros((corners_y*corners_x,3), np.float32)
    objp[:,:2] = np.mgrid[0:corners_x,0:corners_y].T.reshape(-1,2)

    # loop over all calibration images
    for filename in glob.glob(os.path.join('camera_cal','*.jpg')):
        img = cv2.imread(filename)
        # convert to grayscale
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (corners_x, corners_y), None)

        # if we found points, add them to object and image points data
        if ret == True:
            # increase accuracy of corner detection
            cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), corner_finding_termination_criteria)
            obj_points.append(objp)
            img_points.append(corners)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, gray.shape[::-1], None, None)

    if debug_image == True:
        f, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(20, 10))
        ax1.imshow(cv2.cvtColor(cv2.imread("camera_cal/calibration4.jpg"), cv2.COLOR_BGR2GRAY), cmap='gray')
        ax1.set_title('Distorted chessboard')
        ax2.imshow(correct_distortion(cv2.cvtColor(cv2.imread("camera_cal/calibration4.jpg"), cv2.COLOR_BGR2GRAY),
            mtx, dist), cmap='gray')
        ax2.set_title('Undistorted chessboard')
        plt.show()

    return (mtx, dist)

# method for correcting distortion in an image
def correct_distortion(image, mtx, dist):
    return cv2.undistort(image, mtx, dist, None, mtx)

def create_thresholded_binary(image):
    thresholded_binary = np.zeros_like(image)

    # Sobel x gradient - detect horizontal lines
    sobelx = cv2.Sobel(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float), cv2.CV_64F, 1, 0)
    abs_sobelx = np.absolute(sobelx)
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= 40) & (scaled_sobel <= 100)] = 1
    
    # Threshold L channel from LUV color space - detect white lines
    luv = cv2.cvtColor(image, cv2.COLOR_BGR2LUV).astype(np.float)
    l_channel = luv[:,:,0]
    l_binary = np.zeros_like(l_channel)
    l_binary = np.uint8(255*l_binary/np.max(l_binary))
    l_binary[(220 <= l_channel) & (l_channel <= 255)] = 1
    
    # Threshold b channel in Lab color space - detect yellow lines
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab).astype(np.float)
    b_channel = lab[:,:,2]
    b_binary = np.zeros_like(b_channel)
    b_binary = np.uint8(255*b_binary/np.max(b_binary))
    b_binary[(0 <= b_channel) & (b_channel <= 110)] = 1
    
    # Stack each channel to view their individual contributions in green and blue respectively
    # This returns a stack of the two binary images, whose components you can see as different colors
    color_binary = np.dstack(( np.zeros_like(sxbinary), sxbinary, np.zeros_like(sxbinary)))

    # Combine the two binary thresholds
    combined_binary = np.zeros_like(sxbinary)
    combined_binary[(sxbinary == 1) | (l_binary == 1) | (b_binary == 1)] = 1

    return (combined_binary, color_binary)

def compute_perspective_transform_matrices():
    # compute perspective transform based on visual inspection of matching trapezoids
    # in straight_lines1.jpg(pre and post perspective transform)
    src_trap=np.array([[589,455], [692,455], [1039,676], [268,676]], dtype = "float32")
    dst_trap=np.array([[300,0], [1030,0], [980,719], [250,719]], dtype = "float32")
    return (cv2.getPerspectiveTransform(src_trap, dst_trap), cv2.getPerspectiveTransform(dst_trap, src_trap))

# fit a single line polynomial using sliding window technique
def fit_lane_line_polynomial(previous_fit, binary_warped, nonzerox, nonzeroy, x_current, out_img):
    
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    
    # Choose the number of sliding windows
    nwindows = 9
    # Set height of windows
    window_height = np.int(binary_warped.shape[0]/nwindows)

    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50
    
    # Create empty list to receive lane pixel indices
    lane_inds = []
    
    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_x_low = x_current - margin
        win_x_high = x_current + margin
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_x_low,win_y_low),(win_x_high,win_y_high),(0,255,0), 2) 
        # Identify the nonzero pixels in x and y within the window
        good_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_x_low) & (nonzerox < win_x_high)).nonzero()[0]
        # Append these indices to the lists
        lane_inds.append(good_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_inds) > minpix:
            x_current = np.int(np.mean(nonzerox[good_inds]))

    # Concatenate the arrays of indices
    lane_inds = np.concatenate(lane_inds)

    # Extract line pixel positions
    x = nonzerox[lane_inds]
    y = nonzeroy[lane_inds] 

    # Fit a second order polynomial to each
    fit = np.polyfit(y, x, 2)
    
    # Generate x and y values for plotting
    fitx = fit[0]*ploty**2 + fit[1]*ploty + fit[2]

    # Fit new polynomials to x,y in world space for curvature calculation
    fit_cr = np.polyfit(y*ym_per_pix, x*xm_per_pix, 2)
    
    return (fitx, fit, fit_cr, lane_inds)

# fit a single line polynomial using previous fit line restricted search
def fit_lane_line_polynomial_with_previous_fit(binary_warped, previous_fit, nonzerox, nonzeroy, out_img, window_img):
    # margin around previous fit line to search for lane pixels
    margin = 50
    lane_inds = ((nonzerox > (previous_fit[0]*(nonzeroy**2) + previous_fit[1]*nonzeroy + previous_fit[2] - margin)) & (nonzerox < (previous_fit[0]*(nonzeroy**2) + previous_fit[1]*nonzeroy + previous_fit[2] + margin))) 

    # extract line pixel positions
    x = nonzerox[lane_inds]
    y = nonzeroy[lane_inds]

    # if no pixels were found around previous lane line, bail out
    if x.size == 0 or y.size == 0:
        return (None, None, None, None)

    # Fit a second order polynomial
    fit = np.polyfit(y, x, 2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    fitx = fit[0]*ploty**2 + fit[1]*ploty + fit[2]

    # Fit new polynomials to x,y in world space for curvature calculation
    fit_cr = np.polyfit(y*ym_per_pix, x*xm_per_pix, 2)

    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    line_window1 = np.array([np.transpose(np.vstack([fitx-margin, ploty]))])
    line_window2 = np.array([np.flipud(np.transpose(np.vstack([fitx+margin, ploty])))])
    line_pts = np.hstack((line_window1, line_window2))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([line_pts]), (0,255, 0))

    return (fitx, fit, fit_cr, lane_inds)

# fit two lines for both lanes from thresholded binary image
def fit_lane_line_polynomials(binary_warped):
    global previous_left_fit
    global previous_right_fit

    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[np.int(binary_warped.shape[0]/2):,:], axis=0)
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]/2)

    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )

    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    window_img = np.zeros_like(out_img)
    
    # create left and right lane line fits and line positions
    # attempt to fit a line with a restricted search space first
    # if no line is found through that method (or a previous fit line is not available), fall back to a sliding window search
    left_fitx = None
    if previous_left_fit != None:
        left_fitx, left_fit, left_fit_cr, left_lane_inds = fit_lane_line_polynomial_with_previous_fit(binary_warped,
            previous_left_fit, nonzerox, nonzeroy, out_img, window_img)
    
    if left_fitx == None:
        left_fitx, left_fit, left_fit_cr, left_lane_inds = fit_lane_line_polynomial(previous_left_fit, binary_warped, nonzerox,
            nonzeroy, np.argmax(histogram[:midpoint]), out_img)
    previous_left_fit = left_fit

    right_fitx = None
    if previous_right_fit != None:
        right_fitx, right_fit, right_fit_cr, right_lane_inds = fit_lane_line_polynomial_with_previous_fit(binary_warped,
            previous_right_fit, nonzerox, nonzeroy, out_img, window_img)
        
    if right_fitx == None:
        right_fitx, right_fit, right_fit_cr, right_lane_inds = fit_lane_line_polynomial(previous_right_fit, binary_warped, nonzerox,
            nonzeroy, np.argmax(histogram[midpoint:]) + midpoint, out_img)
    previous_right_fit = right_fit
    
    # add window search to output for visualization if desired
    out_img = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)

    return (left_fit_cr, left_fitx, right_fit_cr, right_fitx, out_img, left_lane_inds, right_lane_inds,
        nonzerox, nonzeroy, ploty)

# reset lane identification state between videos / still images
def reset_measurements():
    global previous_left_fit
    global previous_right_fit
    previous_left_fit = None
    previous_right_fit = None
    previous_left_curve_radius.clear()
    previous_right_curve_radius.clear()
    previous_left_fitx.clear()
    previous_right_fitx.clear()

# choose most likely curvature at current image frame
def determine_curve_radius_and_lane_points(image_shape, left_curve_radius, right_curve_radius, left_fitx, right_fitx):
    global previous_left_curve_radius
    global previous_right_curve_radius
    global previous_left_fitx
    global previous_right_fitx
    
    # look at curvatures for past ten frames
    # take average of curvatures which are not are more than 1.5 standard deviations away from the median
    # average remaining curvatures
    # then select pixels for lane display which most closely match that curvature
    num_items = 10
    std_dev_limit = 1.5
    
    if len(previous_left_curve_radius) >= num_items:
        previous_left_curve_radius.popleft()
    previous_left_curve_radius.append(left_curve_radius)

    if len(previous_left_fitx) >= num_items:
        previous_left_fitx.popleft()
    previous_left_fitx.append(left_fitx)
    
    items = np.array(previous_left_curve_radius)
    d = np.abs(items - np.median(items))
    mdev = np.median(d)
    s = d/mdev if mdev else np.array([0])
    valid_items = items[s<std_dev_limit]
    new_left_curve_radius = valid_items.mean()
    
    diff = items - new_left_curve_radius
    new_left_fitx = previous_left_fitx[np.argmin(diff)]
    
    if len(previous_right_curve_radius) >= num_items:
        previous_right_curve_radius.popleft()
    previous_right_curve_radius.append(right_curve_radius)

    if len(previous_right_fitx) >= num_items:
        previous_right_fitx.popleft()
    previous_right_fitx.append(right_fitx)
    
    # take average of curvatures which are not are more than two standard deviations away from the median
    items = np.array(previous_right_curve_radius)
    d = np.abs(items - np.median(items))
    mdev = np.median(d)
    s = d/mdev if mdev else np.array([0])
    valid_items = items[s<std_dev_limit]
    new_right_curve_radius = valid_items.mean()
    
    diff = items - new_right_curve_radius
    new_right_fitx = previous_right_fitx[np.argmin(diff)]

    # assume camera is in the center of the car, which is the midpoint of the image
    image_midpoint = image_shape[1] / 2.0
    lane_midpoint = (new_left_fitx[image_shape[0] - 1] + new_right_fitx[image_shape[0] - 1]) / 2.0
    lane_position = (image_midpoint - lane_midpoint) * xm_per_pix

    return ((new_left_curve_radius + new_right_curve_radius) / 2.0, lane_position, new_left_fitx, new_right_fitx)

# estimate radius of curvature based on chosen
def radius_of_curvature(height, left_fit_cr, right_fit_cr):
    # Define y-value where we want radius of curvature
    left_curverad = ((1 + (2*left_fit_cr[0]*height + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*height + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    return (left_curverad, right_curverad)

# draw detected lane on undistored image
def draw_lines_on_undistorted(image, warped, left_fitx, right_fitx, undist):
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    ploty = np.linspace(0, warped.shape[0]-1, warped.shape[0] )
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, inverse_pespective_transformation_matrix,
        (image.shape[1], image.shape[0]))
 
    # Combine the result with the original image
    return cv2.addWeighted(undist, 1, newwarp, 0.3, 0)

# print out estimated curvature and vehicle position on image
def draw_curvature_and_vehicle_position(image, curve_radius, lane_position):
    mid = cv2.putText(image, "Estimated Lane Curvature Radius: %sm" % int(curve_radius), (50, 50), cv2.FONT_HERSHEY_DUPLEX, 1, 2)
    return cv2.putText(image, "Estimated Lane Position (right of center): %sm" % lane_position, (50, 100), cv2.FONT_HERSHEY_DUPLEX, 1, 2)

# completely process a single BGR image
def process_image(image):
    # Apply a distortion correction to raw images.
    undistorted = correct_distortion(image, calibration_matrix, distortion_coefficients)
    if debug_image == True:
        f, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(20, 10))
        ax1.imshow(image)
        ax1.set_title("Distorted road image")
        ax2.imshow(undistorted)
        ax2.set_title("Undistorted road image")
        plt.show()

    # Use color transforms, gradients, etc., to create a thresholded binary image.
    (thresholded_binary, color_binary) = create_thresholded_binary(undistorted)
    if debug_image == True:
        f, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(20, 10))
        ax1.imshow(undistorted)
        ax1.set_title("Road image")
        ax2.imshow(thresholded_binary)
        ax2.set_title("Thresholded binary")
        plt.show()

    # Apply a perspective transform to rectify binary image ("birds-eye view").
    warped_binary = cv2.warpPerspective(thresholded_binary, pespective_transformation_matrix, tuple(reversed(thresholded_binary.shape)),
        flags=cv2.INTER_LINEAR)
    
    if debug_image == True:
        f, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(20, 10))
        ax1.imshow(undistorted)
        ax1.set_title("Original image")
        ax2.imshow(cv2.warpPerspective(undistorted, pespective_transformation_matrix, tuple(reversed(thresholded_binary.shape)),
            flags=cv2.INTER_LINEAR))
        ax2.set_title("Warped image")
        plt.show()
    
    # Detect lane pixels and fit to find the lane boundary.
    left_fit, left_fitx, right_fit, right_fitx, out_img, left_lane_inds, right_lane_inds, nonzerox, nonzeroy, ploty = fit_lane_line_polynomials(warped_binary)
    if debug_image == True:
        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
        plt.figure(figsize=(20,10))
        plt.imshow(out_img)
        plt.plot(left_fitx, ploty, color='yellow')
        plt.plot(right_fitx, ploty, color='yellow')
        plt.xlim(0, 1280)
        plt.ylim(720, 0)
        plt.show()

    # Determine the curvature of the lane and vehicle position with respect to center.
    left_curve_radius, right_curve_radius = radius_of_curvature(warped_binary.shape[0], left_fit, right_fit)
    
    curve_radius, lane_position, chosen_left_fitx, chosen_right_fitx = determine_curve_radius_and_lane_points(warped_binary.shape, left_curve_radius, right_curve_radius, left_fitx, right_fitx)

    # Warp the detected lane boundaries back onto the original image.
    image_with_lane = draw_lines_on_undistorted(image, warped_binary, chosen_left_fitx, chosen_right_fitx, undistorted)

    # Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.
    final = draw_curvature_and_vehicle_position(image_with_lane, curve_radius, lane_position)
    
    if debug_image == True:
        plt.figure(figsize=(20,10))
        plt.imshow(final)
        plt.show()
    
    return final

# ENTRY POINT

print("computing camera calibration matrix, distortion coefficients, and perspective transform matrices...")

# Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
calibration_matrix, distortion_coefficients = compute_calibration_mtx_and_distortion_coeff()

# Compute perspective transform matrices
pespective_transformation_matrix, inverse_pespective_transformation_matrix = compute_perspective_transform_matrices()

# run image processing on test images
for test_image in glob.glob(os.path.join('test_images','*.jpg')):
    print("Processing %s..." % test_image)
    reset_measurements()
    cv2.imwrite(os.path.join('output_images', os.path.basename(test_image)), cv2.cvtColor(process_image(cv2.cvtColor(cv2.imread(test_image), cv2.COLOR_RGB2BGR)), cv2.COLOR_BGR2RGB))

# run image processing on test videos
for file_name in glob.glob("*.mp4"):
    if "_processed" in file_name:
        continue
    print("Processing %s..." % file_name)
    reset_measurements()
    VideoFileClip(file_name).fl_image(process_image).write_videofile(
        os.path.splitext(file_name)[0] + "_processed.mp4", audio=False)
