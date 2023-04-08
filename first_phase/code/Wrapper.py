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
from FeatureDescriptors import *
from utils.PlotUtils import *
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
    """
	Corner Detection
	Save Corner detection output as corners.png
	"""
    C_imgs = detectCornersHarris(all_imgs, show=False)
    """
	Perform ANMS: Adaptive Non-Maximal Suppression
	Save ANMS output as anms.png
	"""
    ANMS_corners = getBestCorners(C_imgs)
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
    matches_ = FeatureMatching(all_imgs[0], all_imgs[1], ANMS_corners[0], ANMS_corners[1])
    # print(matches_)
    drawMatches(all_imgs[0], all_imgs[1], matches_)
    # an_array = np.array([2, 2, 2, 2])
    # an_array.reshape((len(an_array), -1))
    # another_array = np.array([1, 1, 1, 1])
    # another_array.reshape((len(another_array), -1))
    # ssd = np.sum((an_array - another_array)**2)
    # print(ssd)
    """
	Refine: RANSAC, Estimate Homography
	"""

    """
	Image Warping + Blending
	Save Panorama output as mypano.png
	"""


if __name__ == "__main__":
    main()
