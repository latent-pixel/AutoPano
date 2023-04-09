import numpy as np
import cv2
from utils.MathUtils import *


def getInliers(matches, n_iter = 1000, eps = 10):
    inliers_idx = []
    best_H = None
    for i in range(n_iter):
        random_idx = np.random.choice(len(matches), 4)
        matches_batch = matches[random_idx]
        H = cv2.getPerspectiveTransform(np.float32(matches_batch[:, 0:2]), np.float32(matches_batch[:, 2:4]))
        s = []
        for j in range(len(matches)):
            p1 = np.array([matches[j, 0], matches[j, 1], 1.]).reshape((3, -1))
            p2 = np.array([matches[j, 2], matches[j, 3], 1.]).reshape((3, -1))
            H_p1 = np.dot(H, p1)
            H_p1 = H_p1 / (H_p1[-1]+ 1e-06)
            if np.sqrt(computeSSD(p2, H_p1)) < eps:
                s.append(j)
        if len(inliers_idx) < len(s):
            inliers_idx = s
            best_H = H 

    return inliers_idx, best_H