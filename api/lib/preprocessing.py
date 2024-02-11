import cv2
import numpy as np
from skimage.transform import resize

lower_bound = [90, 50, 50]
upper_bound = [130, 255, 255]


def preprocess_for_feeling(img):
    img_resized = resize(img, (64, 64))
    img_resized = cv2.equalizeHist((img_resized * 255).astype(np.uint8))
    sobelx = cv2.Sobel(np.float32(img_resized), cv2.CV_64F, 1, 0, ksize=9)
    sobely = cv2.Sobel(np.float32(img_resized), cv2.CV_64F, 0, 1, ksize=9)
    sobel = np.hypot(sobelx, sobely)
    max_value = np.max(sobel)
    if max_value == 0:
        max_value = 1e-5
    sobel *= 255.0 / max_value
    return sobel


def preprocess_for_eyes(img):
    bgr_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    #hsv_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
    img_resized = resize(bgr_img, (30, 30))
    img_resized = (img_resized * 255).astype(np.uint8)
    return img_resized
