import os
import numpy as np
import cv2
from config import DATA_PATH
from DM.manage_nii import NiiFileManager

def get_mid_slices(nii, dim, num):
    dim_size = nii.size[dim]
    return nii.get_slice(int((dim_size-num)/2), int((dim_size+num)/2), dim)

def get_image_otsu(img):
    img = (img / img.max() * 255).astype(np.uint8)
    otsu, _ = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return otsu

def get_binary_image(img, threshold):
    _, binary_image = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
    return binary_image

def get_image_center(img):
    c

def main():
    file_name = os.path.join(DATA_PATH, 'PDsample', 'HC', 'T1_rigid.nii.gz')
    nii = NiiFileManager(file_name)
    nii.load()
    s1 = get_mid_slices(nii, 2, 40)

if __name__ == "__main__":
    main()