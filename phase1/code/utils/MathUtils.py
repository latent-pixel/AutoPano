import numpy as np
from skimage.feature import peak_local_max


def getANMSCorners(corner_score_images, N_best=500):
    ANMS_corners = []
    for C_img in corner_score_images:
        coordinates = peak_local_max(C_img, min_distance=8)
        # print(coordinates.shape)
        best_corners = []
        for i in range(coordinates.shape[0]):
            ri = np.inf
            xi, yi = coordinates[i]
            for j in range(coordinates.shape[0]):
                xj, yj = coordinates[j]
                if (xi != xj and yi != yj) and (C_img[xi, yi] < 0.9*C_img[xj, yj]):
                    ED = (xj - xi)**2 + (yj - yi)**2
                    if ED < ri:
                        ri = ED
            best_corners.append([yi, xi, ri])
        best_corners.sort(key=lambda x: x[2], reverse=True)
        best_corners = np.array(best_corners[0:N_best])[:, :2]
        ANMS_corners.append(np.int32(best_corners))
    return ANMS_corners


def computeSSD(vec1, vec2):
    return np.sum((vec1 - vec2)**2)
