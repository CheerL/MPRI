#%%
import os
import random
import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage import measure
from config import DATA_PATH
from DM.manage_nii import NiiFileManager


def get_mid_slices(nii, dim, num):
    dim_size = nii.size[dim]
    return nii.get_slice(int((dim_size-num)/2), int((dim_size+num)/2), dim)

def get_otsu(img, only_otsu=True):
    otsu, otsu_img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if only_otsu:
        return otsu
    return otsu, otsu_img

def get_binary_image(img, threshold, add=0):
    _, binary_image = cv2.threshold(img, threshold + add * 255, 255, cv2.THRESH_BINARY)
    return binary_image

def get_center(img):
    moments = cv2.moments(img)
    center_x = int(moments['m10']/moments['m00'])
    center_y = int(moments['m01']/moments['m00'])
    return center_y, center_x

def get_clahe_image(img, limit, row, col):
    height, width = img.shape
    clahe = cv2.createCLAHE(limit * 255, (int(height/row), int(width/col)))
    return clahe.apply(img)

def get_distance(point_1, point_2):
    return ((point_1[0] - point_2[0]) ** 2 + (point_1[1] - point_2[1]) ** 2) ** (1/2)

def get_connected_component(img, min_area=0, sort=True):
    # label = measure.label(img)
    # regs = [reg for reg in measure.regionprops(label) if reg.area > min_area]
    # if sort:
    #     regs.sort(key=lambda r: r.area, reverse=True)
    # return regs
    _, label, stats, centroids = cv2.connectedComponentsWithStats(img)
    info = [
        (level, stat, centroid)
        for level, (stat, centroid) in enumerate(zip(stats, centroids))
        if stat[4] > min_area and level != 0
    ]
    if sort:
        info.sort(key=lambda x: x[1][4], reverse=True)
    return label, info

def get_area_distance(img, point):
    random_points = list(zip(*np.where(img)))
    distances = [get_distance(random_point, point) for random_point in random_points]
    return np.array(distances).min()

def get_center_component(label, info, rate=1/3):
    height, width = label.shape
    up_limit, left_limit = rate * height, rate * width
    down_limit, right_limit = height - up_limit, width - left_limit
    return [
        (level, stat, centroid)
        for level, stat, centroid in info
        if (
            stat[0] > left_limit and
            stat[1] > up_limit and
            stat[0] + stat[2] < right_limit and
            stat[1] + stat[3] < down_limit
        )
    ]

def get_near_component(label, info, point, limit):
    return [
        (level, stat, centroid)
        for level, stat, centroid in info
        if get_area_distance(label == level, point) < limit
    ]

def get_corpus(img, clahe_limit=0.02, clahe_row=8, clahe_col=8,
               clahe_bin_add=0.2, center_rate=1/4,
               min_area=200, max_distance=45,
               test=False):
               #clahe_bin_add=0.15 for MSA
    _, bin_img = get_otsu(img, False)
    center = get_center(bin_img)
    clahe_img = get_clahe_image(img, clahe_limit, clahe_row, clahe_col)
    clahe_otsu = get_otsu(clahe_img)
    clahe_bin_img = get_binary_image(clahe_img, clahe_otsu, clahe_bin_add)
    label, info = get_connected_component(clahe_bin_img, min_area)
    info = get_center_component(label, info, center_rate)
    info = get_near_component(label, info, center, max_distance)
    try:
        corpus_info = info[0]
        corpus = label == corpus_info[0]
    except IndexError:
        corpus_info = info
        corpus = label

    if test:
        test_data = {
            'bin_img': bin_img,
            'clahe_img': clahe_img,
            'clahe_bin_img': clahe_bin_img,
            'label': label,
            'center': center,
            'info': info
        }
        return corpus, corpus_info test_data
    return corpus, corpus_info

def add_point(img, point, size=5, color=127):
    return cv2.circle(img.copy(), point, size, color, -1)

def show(imgs, dim=0, col=2, size=6):
    if isinstance(imgs, np.ndarray):
        if len(imgs.shape) == 2:
            imgs = [imgs]
            col = 1
        elif len(imgs.shape) == 3:
            imgs = np.rollaxis(imgs, dim) if dim != 0 else imgs
        else:
            raise TypeError()
    assert isinstance(imgs, (list, tuple, np.ndarray))
    img_num = len(imgs)
    row = np.ceil(img_num / col).astype(dtype=np.int32)
    _, axes = plt.subplots(row, col,  figsize=(size * col, size * row))
    for img, ax in zip(imgs, axes.reshape(-1)):
        assert isinstance(img, np.ndarray)
        ax.imshow(img, 'gray')

#%%
if __name__ == "__main__":
    file_name = os.path.join(DATA_PATH, 'PDsample', 'HC', 'T1_rigid.nii.gz')
    nii = NiiFileManager(file_name)
    nii.load()
    s1 = get_mid_slices(nii, 2, 40)
