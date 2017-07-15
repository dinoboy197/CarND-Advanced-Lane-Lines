# Advanced Lane Finding

import cv2
import glob
import numpy as np
import os

def compute_calibration_mtx_and_distortion_coeff():
    # Make a list of calibration images
    corner_finding_termination_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    
    obj_points = []
    img_points = []
    
    corners_x = 9
    corners_y = 6
    objp = np.zeros((6*corners_x,3), np.float32)
    objp[:,:2] = np.mgrid[0:corners_x,0:6].T.reshape(-1,2)
    
    for filename in glob.glob(os.path.join('camera_cal','*.jpg')):
        # read in image data
        img = cv2.imread(filename)
        # convert to grayscale
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (corners_x, corners_y), None)
        
        # if we found points, add them
        if ret == True:
            obj_points.append(objp)
            
            # increase accuracy of corner detection
            cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), corner_finding_termination_criteria)
            img_points.append(corners)
    
        
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, gray.shape[::-1], None, None)

    return (mtx, dist)

def correct_distortion(img, mtx, dist):
    return cv2.undistort(img, mtx, dist, None, mtx)

# Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.

calibration_matrix, distortion_coefficients = compute_calibration_mtx_and_distortion_coeff()

# Apply a distortion correction to raw test images.

for test_image in glob.glob(os.path.join('test_images','*.jpg')):
    img = cv2.imread(test_image)
    dst = correct_distortion(img, calibration_matrix, distortion_coefficients)
    cv2.imwrite(os.path.join('output_images', os.path.basename(test_image)), dst)

# Apply a distortion correction to raw test images.
# Use color transforms, gradients, etc., to create a thresholded binary image.
# Apply a perspective transform to rectify binary image ("birds-eye view").
# Detect lane pixels and fit to find the lane boundary.
# Determine the curvature of the lane and vehicle position with respect to center.
# Warp the detected lane boundaries back onto the original image.
# Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.