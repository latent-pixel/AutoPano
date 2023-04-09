import numpy as np
import cv2
import os
from skimage.feature import peak_local_max


def readImagesFromFolder(imdir, show=False):
    all_images = []
    for file in sorted(os.listdir(imdir)):
        if ".jpg" or ".jpeg" or ".png" in file:
            file_path = os.path.join(imdir, file)
            image = cv2.imread(file_path)
            h, w, _ = image.shape
            # if h > 1000 or w > 1000:
            #     image = cv2.resize(image, (0, 0), fx = 0.2, fy = 0.2)
            all_images.append(image)
            if show:
                cv2.imshow("image", image)
                cv2.waitKey(0)
    cv2.destroyAllWindows()
    return all_images


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


def paddedWarping(src_image, dst_image, transform):
    h, w, c = src_image.shape
    h_dst, w_dst, c_dst = dst_image.shape
    src_homo = np.array([[0, 0, 1], # left-top
                    [h, 0, 1], # left-bottom
                    [h, w, 1], # right-bottom
                    [0, w, 1]]) # right-top
    # src_homo = np.stack((src, np.ones((4, 1))), axis=1)
    transformed_src = transform.dot(src_homo.T)
    transformed_src = np.transpose(transformed_src / transformed_src[2, :])
    x_max, x_min = np.max(transformed_src[:, 0]), np.min(transformed_src[:, 0])
    y_max, y_min = np.max(transformed_src[:, 1]), np.min(transformed_src[:, 1])
    pad_x = np.round(np.maximum(x_max, w_dst) - np.minimum(x_min, 0)).astype(int)
    pad_y = np.round(np.maximum(y_max, h_dst) - np.minimum(y_min, 0)).astype(int)
    padded_shape = (pad_y, pad_x, c_dst)
    dst_padded = np.zeros(padded_shape, dtype=np.uint8)
    
    # computing the updated transform and shifting the destination image
    translation = np.eye(3, 3)
    dst_x_min, dst_y_min = 0, 0
    if x_min < 0:
        translation[0, 2] = np.round(-x_min).astype(int)
        dst_x_min = np.round(-x_min).astype(int)
    if y_min < 0:
        translation[1, 2] += np.round(-y_min).astype(int)
        dst_y_min = np.round(-y_min).astype(int)
    new_transform = translation.dot(transform)  # Important!
    new_transform /= new_transform[2, 2]

    dst_padded[dst_y_min:dst_y_min+h_dst, dst_x_min:dst_x_min+w_dst] = dst_image
    src_warped = cv2.warpPerspective(src_image, new_transform, (pad_x, pad_y))
    return src_warped, dst_padded
