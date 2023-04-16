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


def main():
    # Add any Command Line arguments here
    # Parser = argparse.ArgumentParser()
    # Parser.add_argument('--NumFeatures', default=100, help='Number of best features to extract from each image, Default:100')

    # Args = Parser.parse_args()
    # NumFeatures = Args.NumFeatures
    corner_algs = {"Harris":0, "ShiTomasi":1} 
    corner_detect = corner_algs["ShiTomasi"]
    alpha = 0.5
    all_imgs = readImagesFromFolder("first_phase/data/Train/Set1", show=False)
    # panorama = createPanorama(all_imgs[:5], corner_detect)
    # cv2.imwrite('first_phase/panorama.png', panorama)

    # Load input images
    # all_imgs.reverse()
    # all_imgs = all_imgs[3:]
    # Loop until only one image left in the list
    # while len(all_imgs) > 1:
    #     print(len(all_imgs))
    #     # Select a pair of images
    #     dst_image = all_imgs.pop(0)
    #     src_image = all_imgs.pop(0)
        
    #     # Compute image registration
    #     matches = FeatureMatching(dst_image, src_image, corner_detect)
    #     best_idx, best_H = getInliers(matches)
        
    #     # Warp and blend
    #     pano = paddedWarping(dst_image, src_image, best_H)
    #     pano = pano.astype('uint8')
        
    #     # Update the image list
    #     all_imgs.insert(0, pano)
 
    # # Final panorama
    # final_panorama = all_imgs[0]
    # # cv2.imshow("panorama", final_panorama)
    # # cv2.waitKey(0)
    # # cv2.destroyAllWindows()

    # cv2.imwrite('first_phase/panorama.png', final_panorama)

    test_image = all_imgs[0]
    image_cyl, mask_x, mask_y = projectOntoCylinder(test_image)
    
    # image_mask = np.zeros(image_cyl.shape, dtype=np.uint8)
    # image_mask[mask_y, mask_x, :] = 255
    cv2.imshow("projection", image_cyl)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
