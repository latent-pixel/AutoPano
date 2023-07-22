import numpy as np
import cv2
from utils.ImageUtils import *
from utils.MathUtils import *
from utils.PlotUtils import *
from CornerDetect import *
from FeatureMatching import *
from GetInliersRANSAC import *
from WarpnBlend import *


def stitchImages(src_image, dst_image, f, features=None, corner_detector=None):
    src_image, mask_x, mask_y = projectOntoCylinder(src_image, f)
    dst_image, _, _ = projectOntoCylinder(dst_image, f)
    sel_images = [src_image, dst_image]

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
    
    src_image_mask = np.zeros(src_image.shape, dtype=np.uint8)
    src_image_mask[mask_y, mask_x, :] = 255
    new_frame_size, correction, new_H = getNewFrameSize(src_image, dst_image, best_H)
    src_image_transformed = cv2.warpPerspective(src_image, new_H, (new_frame_size[1], new_frame_size[0]))
    src_image_transformed_mask = cv2.warpPerspective(src_image_mask, new_H, (new_frame_size[1], new_frame_size[0]))
    dst_image_transformed = np.zeros((new_frame_size[0], new_frame_size[1], 3), dtype=np.uint8)
    dst_image_transformed[correction[1]:correction[1]+dst_image.shape[0], correction[0]:correction[0]+dst_image.shape[1], :] = dst_image

    stitched_image = cv2.bitwise_or(src_image_transformed, cv2.bitwise_and(dst_image_transformed, cv2.bitwise_not(src_image_transformed_mask)))

    return stitched_image