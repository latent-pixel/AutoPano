import numpy as np
import cv2
from skimage.feature import peak_local_max


def getBestCorners(corner_score_images, N_best=100):
    ANMS_corners = []
    for C_img in corner_score_images:
        coordinates = peak_local_max(C_img, min_distance=10)
        # print(coordinates.shape)
        best_corners = []
        for i in range(coordinates.shape[0]):
            ri = np.inf
            xi, yi = coordinates[i]
            for j in range(coordinates.shape[0]):
                xj, yj = coordinates[j]
                if (xi != xj and yi != yj) and C_img[xj, yj] > C_img[xi, yi]:
                    ED = np.sqrt((xj - xi)**2 + (yj - yi)**2)
                    if ED < ri:
                        ri = ED
            best_corners.append([yi, xi, ri])
        best_corners.sort(key=lambda x: x[2], reverse=True)
        best_corners = np.array(best_corners[0:N_best])[:, :2]
        ANMS_corners.append(np.int32(best_corners))
    return ANMS_corners
            
