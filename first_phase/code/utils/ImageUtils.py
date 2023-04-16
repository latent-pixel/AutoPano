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


def projectOntoCylinder(image, f=500):

    h, w, c = image.shape
    yc, xc = h // 2, w // 2

    i = np.linspace(0, h-1, h)
    j = np.linspace(0, w-1, w)
    jv, iv  = np.meshgrid(j, i)
    positions = np.vstack([jv.ravel(), iv.ravel()]).T

    y, x = positions[:, 1], positions[:, 0]
    yp = (y - yc) / np.cos ((x - xc) / f) + yc
    xp = f*np.tan((x - xc) / f) + xc
    inlier_idxs = (xp >= 0)*(yp >= 0)*(xp < w-1)*(yp < h-1)
    yp_in, xp_in =  yp[inlier_idxs], xp[inlier_idxs]
    y_in, x_in = y[inlier_idxs].astype(int), x[inlier_idxs].astype(int)
    
    # Bilinear Interpolation
    x0 = np.floor(xp_in).astype(int)
    x1 = x0 + 1
    y0 = np.floor(yp_in).astype(int)
    y1 = y0 + 1

    x0 = np.clip(x0, 0, w-1)
    x1 = np.clip(x1, 0, w-1)
    y0 = np.clip(y0, 0, h-1)
    y1 = np.clip(y1, 0, h-1)

    Ia = image[y0, x0, :]
    Ib = image[y1, x0, :]
    Ic = image[y0, x1, :]
    Id = image[y1, x1, :]

    wa = (x1-xp_in) * (y1-yp_in)
    wb = (x1-xp_in) * (yp_in-y0)
    wc = (xp_in-x0) * (y1-yp_in)
    wd = (xp_in-x0) * (yp_in-y0)

    projection = np.zeros_like(image, dtype=np.uint8)
    projection[y_in, x_in, :] = wa[:, None]*Ia + wb[:, None]*Ib + wc[:, None]*Ic + wd[:, None]*Id
    min_x = min(x_in)
    projection = projection[:, min_x:-min_x, :]

    return projection, x_in-min_x, y_in
