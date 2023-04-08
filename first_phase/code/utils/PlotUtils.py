import numpy as np
import cv2


def drawMatches(image1, image2, matches_list):
    w, h, c = image1.shape
    joined_image = np.concatenate((image1, image2), axis=1)
    image1_pts = matches_list[:, 0:2].astype(int)
    image2_pts = matches_list[:, 2:4].astype(int)
    for i in range(image1_pts.shape[0]):
        pt_image1 = (image1_pts[i, 0], image1_pts[i, 1])
        pt_image2 = (h+image2_pts[i, 0], image2_pts[i, 1])
        joined_image = cv2.circle(joined_image, pt_image1, radius=0, color=(0, 255, 0), thickness=5)
        joined_image = cv2.circle(joined_image, pt_image2, radius=0, color=(0, 255, 0), thickness=5)
        joined_image = cv2.line(joined_image, pt_image1, pt_image2, color=(0, 255, 0), thickness=1)
    cv2.imshow("matches", joined_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()