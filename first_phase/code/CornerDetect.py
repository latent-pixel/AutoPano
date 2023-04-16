import numpy as np
import cv2
import os


def detectCornersHarris(images, show=False, save=False, save_path="../results/"):
    corner_score_images = []
    for idx, image in enumerate(images):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = np.float32(gray)
        block_size, ksize, k = 2, 3, 0.04
        dst = cv2.cornerHarris(gray, block_size, ksize, k)
        dst[dst < 0.01*dst.max()] = 0     # Check!
        corner_score_images.append(dst)
        if show:
            temp_image = image.copy()
            temp_dst = dst.copy()
            temp_dst = cv2.dilate(temp_dst, None)
            temp_image[temp_dst > 0.01*temp_dst.max()] = [0, 0, 255]
            cv2.imshow("Corners score image", temp_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            if save:
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                    cv2.imwrite(save_path+"/HarrisCorners{}.jpg".format(idx), temp_image)
    return corner_score_images


def detectCornersShiTomasi(images, show=False):
    all_corners = []
    for image in images:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        corners = cv2.goodFeaturesToTrack(gray, 1000, 0.01, 10)
        corners = np.int32(corners)
        all_corners.append(corners.squeeze())
        if show:
            for i in corners:
                x, y = i.ravel()
                cv2.circle(image, (x, y), 3, (0, 0, 255), -1)
            cv2.imshow("Corners Image", image)
            cv2.waitKey()
            cv2.destroyAllWindows()
    return all_corners