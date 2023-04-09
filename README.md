# My AutoPano

In this project, we stitch images together using traditional feature matching and deep learning methods. The `first_phase` folder contains the implementation for feature matching method and the deep learning method is a work in progress.

## **Phase-1: Feature Matching**

**1.1. Identify corners in the images using corner detection algorithms like `Harris` or `Shi-Tomasi`.**
    
**1.2. Apply Adaptive Non-Maximal Suppression (ANMS) on the detected corners to ensure that the corners are evenly spread out.**

**1.3. Find the feature descriptors (feature vectors) corresponding to each of the ANMS corners.**

**1.4. Match these feature vectors based on the sum of squared distances between the vectors (smaller is better).**

**1.5. Filter outliers using RANSAC on the obtained feature matches and compute the homography using the inliers.**

**1.6. Warp and blend the images.**