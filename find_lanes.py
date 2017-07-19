# Advanced Lane Finding

import cv2
import glob
import numpy as np
import os
from moviepy.editor import VideoFileClip
import matplotlib.pyplot as plt

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

    return (mtx, dist)

# method for correcting distortion in an image
def correct_distortion(image, mtx, dist):
    return cv2.undistort(image, mtx, dist, None, mtx)

def create_thresholded_binary(image):
    thresholded_binary = np.zeros_like(image)
    hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS).astype(np.float)
    s_channel = hls[:,:,2]

    # Sobel gradient finding in x direction
    sobelx = cv2.Sobel(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float), cv2.CV_64F, 1, 0)
     # Absolute x derivative to accentuate lines away from horizontal
    abs_sobelx = np.absolute(sobelx)
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    
    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= 20) & (scaled_sobel <= 100)] = 1
    
    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= 170) & (s_channel <= 255)] = 1

    # Combine the two binary thresholds
    combined_binary = np.zeros_like(sxbinary)
    combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1

    return combined_binary

def compute_perspective_transform_matrices():
    # compute perspective transform based on visual inspection of matching trapezoids
    # in straight_lines1.jpg(pre and post perspective transform)
    src_trap=np.array([[597,448], [685,448], [1039,676], [268,676]], dtype = "float32")
    dst_trap=np.array([[300,0], [1030,0], [980,719], [250,719]], dtype = "float32")
    return (cv2.getPerspectiveTransform(src_trap, dst_trap), cv2.getPerspectiveTransform(dst_trap, src_trap))

def fit_lane_line_polynomials(binary_warped):
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[np.int(binary_warped.shape[0]/2):,:], axis=0)
    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    
    # Choose the number of sliding windows
    nwindows = 9
    # Set height of windows
    window_height = np.int(binary_warped.shape[0]/nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []
    
    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2) 
        cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2) 
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
    
    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)
    
    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    
    # Generate x and y values for plotting
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    ym_per_pix = 30/720 # meters per pixel in y dimension (given by input dataset authors)
    xm_per_pix = 3.7/700 # meters per pixel in x dimension (given by input dataset authors)

    # Fit new polynomials to x,y in world space for curvature calculation
    left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)

    #out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    #out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
    #plt.imshow(out_img)
    #plt.plot(left_fitx, ploty, color='yellow')
    #plt.plot(right_fitx, ploty, color='yellow')
    #plt.xlim(0, 1280)
    #plt.ylim(720, 0)
    #plt.show()
    
    return (left_fit_cr, left_fitx, right_fit_cr, right_fitx)

def radius_of_curvature(height, left_fit_cr, right_fit_cr):
    # Define y-value where we want radius of curvature
    left_curverad = ((1 + (2*left_fit_cr[0]*height + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*height + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    return (left_curverad, right_curverad)

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

def process_image(image):
    # Apply a distortion correction to raw images.
    undistorted = correct_distortion(image, calibration_matrix, distortion_coefficients)

    # Use color transforms, gradients, etc., to create a thresholded binary image.
    thresholded_binary = create_thresholded_binary(undistorted)

    # Apply a perspective transform to rectify binary image ("birds-eye view").
    warped_binary = cv2.warpPerspective(thresholded_binary, pespective_transformation_matrix, tuple(reversed(thresholded_binary.shape)),
        flags=cv2.INTER_LINEAR)
    
    # Detect lane pixels and fit to find the lane boundary.
    left_fit, left_fitx, right_fit, right_fitx = fit_lane_line_polynomials(warped_binary)
    
    # Determine the curvature of the lane and vehicle position with respect to center.
    left_curve_radius, right_curve_radius = radius_of_curvature(warped_binary.shape[0], left_fit, right_fit)

    # Warp the detected lane boundaries back onto the original image.
    final = draw_lines_on_undistorted(image, warped_binary, left_fitx, right_fitx, undistorted)
    
    # Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.
    #print(left_curve_radius, 'm', right_curve_radius, 'm')
    
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
    cv2.imwrite(os.path.join('output_images', os.path.basename(test_image)), process_image(cv2.imread(test_image)))

# run image processing on test videos
#for file_name in glob.glob("*.mp4"):
#    if "_processed" in file_name:
#        continue
#    print("Processing %s..." % file_name)
#    VideoFileClip(file_name).fl_image(process_image).write_videofile(
#        os.path.splitext(file_name)[0] + "_processed.mp4", audio=False)
