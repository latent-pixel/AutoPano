#!/usr/bin/evn python

"""
Author(s):
Bhargav Kumar Soothram (bsoothra@umd.edu)
M.Eng. in Robotics,
University of Maryland
"""

# Code starts here:

import numpy as np
import cv2
from utils.ImageUtils import *
from utils.MathUtils import *
from utils.PlotUtils import *
from FeatureDescriptors import *
from GetInliersRANSAC import *
# Add any python libraries here


def main():
    # Add any Command Line arguments here
    # Parser = argparse.ArgumentParser()
    # Parser.add_argument('--NumFeatures', default=100, help='Number of best features to extract from each image, Default:100')

    # Args = Parser.parse_args()
    # NumFeatures = Args.NumFeatures

    """
    Read a set of images for Panorama stitching
    """
    all_imgs = readImagesFromFolder("first_phase/data/Train/Set1", show=False)
    img1, img2 = all_imgs[0], all_imgs[1]
    """
	Corner Detection
	Save Corner detection output as corners.png
	"""
    corner_algs = {"Harris":0, "ShiTomasi":1} 
    corner_detect = corner_algs["Harris"]
    # C_imgs = detectCornersHarris(all_imgs, show=False)
    """
	Perform ANMS: Adaptive Non-Maximal Suppression
	Save ANMS output as anms.png
	"""
    # ANMS_corners = detectCornersShiTomasi(all_imgs)
    # print(ANMS_corners[0].squeeze())
    # for i in range(len(ANMS_corners)):
    #     temp_img = all_imgs[i].copy()
    #     for coord in ANMS_corners[i]:
    #         x, y = coord.ravel()
    #         cv2.circle(temp_img, (x, y), 3, (0, 0, 255), -1)
    #     cv2.imshow("Corners Image", temp_img)
    #     cv2.waitKey()
    """
	Feature Descriptors
	Save Feature Descriptor output as FD.png
	"""
    # feature_descriptors = getFeatureDescriptors(all_imgs, ANMS_corners)
    """
	Feature Matching
	Save Feature Matching output as matching.png
	"""
    matches_ = FeatureMatching(img1, img2, corner_detector=corner_detect)
    drawMatches(img1, img2, matches_)
    best_idx, best_H = getInliers(matches_)
    best_matches = matches_[best_idx]
    drawMatches(img1, img2, best_matches)
    """
	Refine: RANSAC, Estimate Homography
	"""

    """
	Image Warping + Blending
	Save Panorama output as mypano.png
	"""


if __name__ == "__main__":
    main()
