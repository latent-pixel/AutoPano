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
from FeatureMatching import *
from GetInliersRANSAC import *
from WarpnBlend import *
from StitchImages import stitchImages


def stitchTwoImages(image1, image2, features=None, corner_detector=None):
    
    sel_images = [image1, image2]

    if features == 0:
        if corner_detector == 0:
            C_imgs = detectCornersHarris(sel_images)
            anms_corners = getANMSCorners(C_imgs)
        else:
            anms_corners = detectCornersShiTomasi(sel_images)
        matches = FeatureMatching(sel_images, anms_corners)
        print("    Total matches found:", matches.shape[0])
        best_idx, _ = getInliers(matches, n_iter=1500, eps=10)
        best_matches = matches[best_idx]
        print("    Number of matches after RANSAC:", best_matches.shape[0])
        best_H, mask = cv2.findHomography(best_matches[:, :2], best_matches[:, 2:4])
        pano = paddedWarping(image1, image2, best_H)
        pano = pano.astype('uint8')

    if features == 1:
        keypts1, keypts2, matches = siftFeatureMatching(sel_images, show=False)
        reshaped_matches = []
        for match in matches:
            m11, m12 = keypts1[match.queryIdx].pt
            m21, m22 = keypts2[match.trainIdx].pt
            reshaped_matches.append([m11, m12, m21, m22])
        reshaped_matches = np.array(reshaped_matches, dtype=np.int32)
        best_idx, _ = getInliers(reshaped_matches, n_iter=1500, eps=5)
        best_matches = reshaped_matches[best_idx]
        print("Number of inlier matches:", best_matches.shape[0])
        # drawMatches(image1, image2, best_matches)
        best_H, mask = cv2.findHomography(best_matches[:, :2], best_matches[:, 2:4], cv2.RANSAC)
        pano = paddedWarping(image1, image2, best_H)
        pano = pano.astype('uint8')

    return pano


def main():

    # Parser = argparse.ArgumentParser()
    # Parser.add_argument('--NumFeatures', default=100, help='Number of best features to extract from each image, Default:100')

    # Args = Parser.parse_args()
    # NumFeatures = Args.NumFeatures
    corner_algs = {"Harris":0, "ShiTomasi":1}
    corner_detect = None
    feature_detector = {"patch":0, "sift":1}
    feat = feature_detector["patch"]
    if feat == 0:
        corner_detect = corner_algs["Harris"]

    f = 1000
    all_imgs = readImagesFromFolder("phase1/data/Train/Set3", show=False)

    # imgs_harris = detectCornersHarris(all_imgs, save=True)
    # imgs_ShiTomasi = detectCornersShiTomasi(all_imgs, save=True)

    # Loop until only one image left in the list
    # all_imgs = all_imgs[:2]
    N = len(all_imgs)
    if N > 3:
        left = all_imgs[: N // 2]
        while len(left) > 1:
            print("evaluating left pano", len(left))
            # Select a pair of images
            dst_image = left.pop(0)
            src_image = left.pop(0)

            # dst_image, _, _ = projectOntoCylinder(dst_image, f)
            # src_image, _, _ = projectOntoCylinder(src_image, f)
            
            pano = stitchImages(dst_image, src_image, f, feat, corner_detect) 
            left.insert(0, pano) # Update the image list
        left_panorama = left[0]
        
        right = all_imgs[N//2 :]
        right.reverse()
        while len(right) > 1:
            print("evaluating right pano", len(right))
            # Select a pair of images
            dst_image = right.pop(0)
            src_image = right.pop(0)

            # dst_image, _, _ = projectOntoCylinder(dst_image, f)
            # src_image, _, _ = projectOntoCylinder(src_image, f)

            pano = stitchImages(dst_image, src_image, f, feat, corner_detect)
            right.insert(0, pano)
        right_panorama = right[0]

        final_panorama = stitchImages(left_panorama, right_panorama, f, feat, corner_detect)  # Final panorama

        cv2.imwrite('phase1/testing_panorama.png', final_panorama) 

    else:
        while len(all_imgs) > 1:
            print(len(all_imgs))
            # Select a pair of images
            dst_image = all_imgs.pop(0)
            src_image = all_imgs.pop(0)

            # dst_image, _, _ = projectOntoCylinder(dst_image, f)
            # src_image, _, _ = projectOntoCylinder(src_image, f)

            pano = stitchImages(src_image, dst_image, f, feat, corner_detect)
            # pano = stitchTwoImages(dst_image, src_image, feat, corner_detect)

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
