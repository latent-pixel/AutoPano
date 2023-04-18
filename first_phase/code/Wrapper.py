#!/usr/bin/evn python

"""
Author(s):
Bhargav Kumar Soothram (bsoothra@umd.edu)
M.Eng. Robotics,
University of Maryland
"""


import numpy as np
import cv2
from utils.ImageUtils import *
from utils.MathUtils import *
from utils.PlotUtils import *
from CornerDetect import *
from FeatureDescriptors import *
from GetInliersRANSAC import *
from WarpnBlend import *


def stitchTwoImages(image1, image2, corner_detector):
    matches = FeatureMatching(image1, image2, corner_detector)
    best_idx, _ = getInliers(matches, n_iter=1500, eps=10)
    best_matches = matches[best_idx]
    print(best_matches.shape)
    best_H, mask = cv2.findHomography(best_matches[:, :2], best_matches[:, 2:4])
    pano = paddedWarping(image1, image2, best_H)
    pano = pano.astype('uint8')
    return pano


def main():
    # Add any Command Line arguments here
    # Parser = argparse.ArgumentParser()
    # Parser.add_argument('--NumFeatures', default=100, help='Number of best features to extract from each image, Default:100')

    # Args = Parser.parse_args()
    # NumFeatures = Args.NumFeatures
    corner_algs = {"Harris":0, "ShiTomasi":1} 
    corner_detect = corner_algs["ShiTomasi"]
    f = 800
    alpha = 0.5
    all_imgs = readImagesFromFolder("first_phase/data/Test", show=False)

    # Loop until only one image left in the list
    # all_imgs = all_imgs[:3]
    N = len(all_imgs)
    if N > 3:
        left = all_imgs[: N // 2]
        while len(left) > 1:
            print("evaluating left pano", len(left))
            # Select a pair of images
            dst_image = left.pop(0)
            src_image = left.pop(0)

            dst_image, _, _ = projectOntoCylinder(dst_image, f)
            src_image, _, _ = projectOntoCylinder(src_image, f)
            
            pano = stitchTwoImages(dst_image, src_image, corner_detect) 
            left.insert(0, pano) # Update the image list
        left_panorama = left[0]
        
        right = all_imgs[N//2 :]
        # right.reverse()
        while len(right) > 1:
            print("evaluating right pano", len(right))
            # Select a pair of images
            dst_image = right.pop(0)
            src_image = right.pop(0)

            dst_image, _, _ = projectOntoCylinder(dst_image, f)
            src_image, _, _ = projectOntoCylinder(src_image, f)

            pano = stitchTwoImages(dst_image, src_image, corner_detect)
            right.insert(0, pano)
        right_panorama = right[0]

        final_panorama = stitchTwoImages(left_panorama, right_panorama, corner_detect)  # Final panorama
        # cv2.imshow("panorama", final_panorama)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        cv2.imwrite('first_phase/testing_panorama.png', final_panorama)
        

    else:
        while len(all_imgs) > 1:
            print(len(all_imgs))
            # Select a pair of images
            dst_image = all_imgs.pop(0)
            src_image = all_imgs.pop(0)

            # dst_image, _, _ = projectOntoCylinder(dst_image, f)
            # src_image, _, _ = projectOntoCylinder(src_image, f)

            pano = stitchTwoImages(dst_image, src_image, corner_detect)

            # Update the image list
            all_imgs.insert(0, pano)
        
        # Final panorama
        final_panorama = all_imgs[0]
        cv2.imshow("panorama", final_panorama)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # cv2.imwrite('first_phase/testing_panorama.png', final_panorama)


if __name__ == "__main__":
    main()
