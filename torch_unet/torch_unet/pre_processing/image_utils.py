import gc

import cv2
import numpy as np


def rotate_image(img, angle):
    # Rotates an image given an angle
    height, width = img.shape[:2]
    image_center = (
        width / 2, height / 2)  # getRotationMatrix2D needs coordinates in reverse order (width, height) compared to shape
    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.)
    abs_cos = abs(rotation_mat[0, 0])
    abs_sin = abs(rotation_mat[0, 1])
    
    # find the new width and height bounds
    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)
    
    # subtract old image center (bringing image back to origo) and adding the new image center coordinates
    rotation_mat[0, 2] += bound_w / 2 - image_center[0]
    rotation_mat[1, 2] += bound_h / 2 - image_center[1]
    
    # rotate image with the new bounds and translated rotation matrix
    rotated_mat = cv2.warpAffine(img, rotation_mat, (bound_w, bound_h))
    return rotated_mat


def crop_image(img, h, w):
    # Crops the center of the image given width and height
    height, width = img.shape[:2]
    center_x, center_y = (int(width / 2), int(width / 2))
    bounds_x = (int(center_x - w / 2), int(center_x + w / 2))
    bounds_y = (int(center_y - h / 2), int(center_y + h / 2))
    if len(img.shape) == 2:
        return img[bounds_y[0]: bounds_y[1], bounds_x[0]:bounds_x[1]]
    
    return img[bounds_y[0]: bounds_y[1], bounds_x[0]:bounds_x[1], :]


def mirror_image(img, n):
    # Mirror image with given padding
    if len(img.shape) == 3:
        return np.pad(img, ((n, n), (n, n), (0, 0)), mode="symmetric")
    else:
        return np.pad(img, ((n, n), (n, n)), mode="symmetric")


def rotate_and_crop(img, angle, padding, patch_size):
    # Mirrors the image, then rotates at given angle, then crops to the patch size
    img = mirror_image(img, padding)
    img = rotate_image(img, angle)
    img = crop_image(img, patch_size, patch_size)
    gc.collect()
    return img
