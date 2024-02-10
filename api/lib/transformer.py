import cv2
import numpy as np
from skimage.transform import resize


def transform_for_feeling(img):
    img_resized = resize(img, (64, 64))

    # Increase contrast
    img_resized = cv2.equalizeHist((img_resized * 255).astype(np.uint8))

    # Apply Sobel filter
    sobelx = cv2.Sobel(np.float32(img_resized), cv2.CV_64F, 1, 0, ksize=9)
    sobely = cv2.Sobel(np.float32(img_resized), cv2.CV_64F, 0, 1, ksize=9)
    sobel = np.hypot(sobelx, sobely)
    max_value = np.max(sobel)
    if max_value == 0:
        max_value = 1e-5  # small constant
    sobel *= 255.0 / max_value
    return sobel
