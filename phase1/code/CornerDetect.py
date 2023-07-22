import numpy as np
import cv2
import os


def detectCornersHarris(images, save=False, save_path="first_phase/results/"):
    corner_score_images = []
    for idx, image in enumerate(images):
        temp = image.copy()
        gray = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)
        gray = np.float32(gray)
        block_size, ksize, k = 2, 3, 0.04
        dst = cv2.cornerHarris(gray, block_size, ksize, k)
        dst[dst < 0.001*dst.max()] = 0     # Check!
        corner_score_images.append(dst)
        if save:
            temp_dst = dst.copy()
            temp_dst = cv2.dilate(temp_dst, None)
            temp[temp_dst > 0.01*temp_dst.max()] = [0, 0, 255]
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            cv2.imwrite(save_path+"/HarrisCorners{}.jpg".format(idx+1), temp)
    return corner_score_images


def detectCornersShiTomasi(images, save=False, save_path="first_phase/results/"):
    all_corners = []
    for idx, image in enumerate(images):
        temp = image.copy()
        gray = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)
        dst = cv2.goodFeaturesToTrack(gray, 1000, 0.001, 15)
        dst = np.int32(dst)
        all_corners.append(dst.squeeze())
        if save:
            for i in dst:
                x, y = i.ravel()
                temp = cv2.circle(temp, (x, y), 3, (0, 0, 255), -1)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            cv2.imwrite(save_path+"/STCorners{}.jpg".format(idx+1), temp)
    return all_corners