import numpy as np
import cv2
from FeatureDescriptors import *


def paddedWarping(src_image, dst_image, transform):
    h, w, c = src_image.shape
    h_dst, w_dst, c_dst = dst_image.shape
    src_homo = np.array([[0, 0, 1], # left-top
                        [0, h, 1], # left-bottom
                        [w, h, 1], # right-bottom
                        [w, 0, 1]]) # right-top
    transformed_src = transform.dot(src_homo.T)
    transformed_src = np.transpose(transformed_src / transformed_src[2, :])
    x_max, x_min = max(transformed_src[:, 0]), min(transformed_src[:, 0])
    y_max, y_min = max(transformed_src[:, 1]), min(transformed_src[:, 1])
    pad_x = np.ceil(max(x_max, w_dst) - min(x_min, 0)).astype(int)   # check!
    pad_y = np.ceil(max(y_max, h_dst) - min(y_min, 0)).astype(int)
    padded_shape = (pad_y, pad_x, c_dst)
    dst_padded = np.zeros(padded_shape, dtype=np.uint8)
    
    # computing the updated transform and shifting the destination image
    translation = np.eye(3, 3)
    dst_x_min, dst_y_min = 0, 0
    if x_min < 0:
        translation[0, 2] = np.round(-x_min).astype(int)    # change this to np.floor
        dst_x_min = np.round(-x_min).astype(int)
    if y_min < 0:
        translation[1, 2] = np.round(-y_min).astype(int)
        dst_y_min = np.round(-y_min).astype(int)
    new_transform = translation.dot(transform)  # Important!
    new_transform /= new_transform[2, 2]

    # dst_padded[dst_y_min:dst_y_min+h_dst, dst_x_min:dst_x_min+w_dst] = dst_image
    warped_result = cv2.warpPerspective(src_image, new_transform, (pad_x, pad_y))
    warped_result[dst_y_min:dst_y_min+h_dst, dst_x_min:dst_x_min+w_dst] = dst_image

    return warped_result



