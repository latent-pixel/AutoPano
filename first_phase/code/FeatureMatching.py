import numpy as np
import cv2
from utils.MathUtils import *


def siftFeatureMatching(image_pair, show=False):
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(cv2.cvtColor(image_pair[0], cv2.COLOR_BGR2GRAY), None)
    kp2, des2 = sift.detectAndCompute(cv2.cvtColor(image_pair[1], cv2.COLOR_BGR2GRAY), None)
    bf_matcher = cv2.BFMatcher()
    matches = bf_matcher.knnMatch(des1, des2, k=2)
    good_matches = []
    for i, j in matches:
        if i.distance / j.distance < 0.75:
            good_matches.append(i)
    if show:
        matches_img = cv2.drawMatches(image_pair[0], kp1, image_pair[1], kp2, good_matches, None)
        cv2.imshow("SIFT Matches", matches_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return kp1, kp2, good_matches


def FeatureMatching(image_pair, corners):
    features1, features2 = getFeatureDescriptors(image_pair, corners)
    matches = []
    for i in range(len(features1)):
        SSD = []
        for j in range(len(features2)):
            ssd = computeSSD(features1[i], features2[j])
            SSD.append([ssd, j])
        SSD.sort(key = lambda x: x[0])
        match_ratio = SSD[0][0]/SSD[1][0]
        if match_ratio < 0.75:
            j_sel = SSD[0][1]
            matches.append([corners[0][i][0], corners[0][i][1], corners[1][j_sel][0], corners[1][j_sel][1]])
    return np.array(matches)


def getFeatureDescriptors(images, all_corners):
    all_normalized_vectors = []
    for i in range(len(images)):
        gray = cv2.cvtColor(images[i], cv2.COLOR_BGR2GRAY)
        image_patches = extractImagePatches(gray, all_corners[i])
        normalized_vectors = []
        for patch in image_patches:
            # cv2.imshow("Image Patch", patch)
            # cv2.waitKey(0)
            blurred_patch = cv2.GaussianBlur(patch, (3, 3), 0)
            resized_patch = cv2.resize(blurred_patch, None, fx=0.2, fy=0.2, interpolation = cv2.INTER_CUBIC)    # check image sub-sampling
            patch_vector = np.reshape(resized_patch, (64, 1))
            normalized_vector = (patch_vector - patch_vector.mean()) / patch_vector.std()
            normalized_vectors.append(normalized_vector)
        all_normalized_vectors.append(normalized_vectors)
    # cv2.destroyAllWindows()
    return all_normalized_vectors


def extractImagePatches(gray_image, centers, size=(40, 40)):
    padded_image = np.pad(gray_image, ((int(size[0]/2), ), (int(size[1]/2), )), 'constant')
    corner_patches = []
    for center in centers:
        x, y = center[1], center[0]
        xp, yp = x+int(size[0]/2), y+int(size[0]/2)
        x1, x2 = int(xp-size[1]/2), int(xp+size[1]/2)
        y1, y2 = int(yp-size[0]/2), int(yp+size[0]/2)
        corner_patch = padded_image[x1:x2, y1:y2]
        corner_patches.append(corner_patch)
    return corner_patches