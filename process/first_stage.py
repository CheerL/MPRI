from __future__ import division
from typing import Tuple

import numpy as np

import component
import process
from config import Box, Point
from DM.file_manage import LabelNiiFileManager, RotatedNiiFileManager


def get_mid_num(
    label_nii: LabelNiiFileManager, rate: float=0.06,
    midbrain_label: int=25
) -> int:
    size = label_nii.size[2]
    min_volum = 500
    for num, label in enumerate(label_nii.get_slice(
        start=int((1-rate)*size/2),
        end=int((1+rate)*size/2),
        dim=2
    )):
        volum = (label == midbrain_label).sum()
        if volum < min_volum:
            min_volum = volum
            mid_num = int((1-rate)*size/2) + num

    return mid_num

def get_quad_seg_point(
    image: np.ndarray, label: np.ndarray,
    midbrain_label: int=25, rate: float=1.21,
    box: Box=((-15, 6), (-8, 8)), debug: bool=False
) -> Point:
    assert rate > 1
    right_bound = process.get_bound_point(label==midbrain_label, 'r')
    box_index = (
        slice(right_bound[0]+box[0][0], right_bound[0]+box[0][1]),
        slice(right_bound[1]+box[1][0], right_bound[1]+box[1][1])
    )
    otsu = process.get_otsu(image[box_index])
    bin_image = image > otsu * rate
    components, _ = component.get_connected_component(bin_image[box_index])
    components = sorted(
        [component for component in components if not component.img[box[0][1] - box[0][0] - 1].any()],
        key=lambda component: process.get_area_distance(
            component.img,
            (-box[0][0], -box[1][0])
        )
    )
    if debug:
        process.show([bin_image] + components)
    if not components:
        return get_quad_seg_point(image, label, midbrain_label, rate+0.02, box, debug)
    if components[0].img.sum() < 8:
        return get_quad_seg_point(image, label, midbrain_label, rate-0.02, box, debug)
    boxed_quad_seg_point = components[0].get_bound_point('d')
    quad_seg_point = (
        boxed_quad_seg_point[0]+right_bound[0]+box[0][0],
        boxed_quad_seg_point[1]+right_bound[1]+box[1][0]
    )
    return quad_seg_point

# def get_brainstem_seg_point(label, quad_seg_point, midbrain_label=25, pons_label=26):
#     box = ((-20, 20), (-50, -10))
#     for y in range(quad_seg_point[0]+box[0][0], quad_seg_point[0]+box[0][1]):
#         for x in range(quad_seg_point[1]+box[1][0], quad_seg_point[1]+box[1][1]):
#             block = label[y-1:y+2, x-1:x+2]
#             if not label[y, x] and midbrain_label in block and pons_label in block:
#                 brainstem_seg_point = (y, x)
#     return brainstem_seg_point

def get_pons_area(label: np.ndarray, pons_label: int=26) -> int:
    return (label==pons_label).sum()

def get_midbrain_area(label: np.ndarray, midbrain_label: int=25) -> int:
    return (label==midbrain_label).sum()

def run(
    image_nii: RotatedNiiFileManager, label_nii: LabelNiiFileManager,
    mid_num_rate: float=0.06, quad_seg_rate: float=1.21,
    midbrain_label: int=25, pons_label: int=26,
    box: Box=((-15, 6), (-8, 8)), debug: bool=False
) -> Tuple[Point, int, int, int]:
    mid_num = get_mid_num(
        label_nii, rate=mid_num_rate, midbrain_label=midbrain_label,
    )
    mid_image = image_nii.get_slice(mid_num, dim=2)
    mid_label = label_nii.get_slice(mid_num, dim=2)
    quad_seg_point = get_quad_seg_point(
        mid_image, mid_label, rate=quad_seg_rate,
        midbrain_label=midbrain_label, box=box, debug=debug
    )
    midbrain_area = get_midbrain_area(mid_label, midbrain_label=midbrain_label)
    pons_area = get_pons_area(mid_label, pons_label=pons_label)
    return quad_seg_point, mid_num, pons_area, midbrain_area

def show(
    image_nii: RotatedNiiFileManager, quad_seg_point: Point,
    mid_num: int, point_color: int=255
) -> None:
    process.show(
        process.add_points(
            image_nii.get_slice(mid_num, dim=2),
            [quad_seg_point], 1, point_color
        )
    )
