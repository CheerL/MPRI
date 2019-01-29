import numpy as np

import component
import process


def get_mid_num(label_nii, rate=0.2, left_ventricle_label=8, right_ventricle_label=9):
    size = label_nii.size[2]
    min_volum = 500
    for num, label in enumerate(label_nii.get_slice(
        start=int((1-rate)*size/2),
        end=int((1+rate)*size/2),
        dim=2
    )):
        volum = np.logical_or(
            label==left_ventricle_label,
            label==right_ventricle_label
        ).sum()
        if volum < min_volum:
            min_volum = volum
            mid_num = int((1-rate)*size/2) + num

    return mid_num

def get_quad_seg_point(image, label, pons_label=25, rate=1.2):
    box = ((-20, 10), (-10, 10))
    right_bound = process.get_bound_point(label==pons_label, 'r')
    box_index = (
        slice(right_bound[0]+box[0][0], right_bound[0]+box[0][1]),
        slice(right_bound[1]+box[1][0], right_bound[1]+box[1][1])
    )
    otsu = process.get_otsu(image[box_index])
    bin_image = image > otsu*rate
    components, _ = component.get_connected_component(bin_image[box_index])
    components = sorted(
        [component for component in components],
        key=lambda component: process.get_area_distance(
            component.img,
            (-box[0][0], -box[1][0])
        )
    )
    boxed_quad_seg_point = components[1].get_bound_point('d')
    quad_seg_point = (
        boxed_quad_seg_point[0]+right_bound[0]+box[0][0],
        boxed_quad_seg_point[1]+right_bound[1]+box[1][0]
    )
    return quad_seg_point

# def get_brainstem_seg_point(label, quad_seg_point, pons_label=25, midbrain_label=26):
#     box = ((-20, 20), (-50, -10))
#     for y in range(quad_seg_point[0]+box[0][0], quad_seg_point[0]+box[0][1]):
#         for x in range(quad_seg_point[1]+box[1][0], quad_seg_point[1]+box[1][1]):
#             block = label[y-1:y+2, x-1:x+2]
#             if not label[y, x] and pons_label in block and midbrain_label in block:
#                 brainstem_seg_point = (y, x)
#     return brainstem_seg_point

def get_midbrain_area(label, midbrain_label=26):
    return (label==midbrain_label).sum()

def get_pons_area(label, pons_label=25):
    return (label==pons_label).sum()

def run(
    image_nii, label_nii, mid_num_rate=0.2, quad_seg_rate=1.2,
    left_ventricle_label=8, right_ventricle_label=9,
    pons_label=25, midbrain_label=26
):
    mid_num = get_mid_num(
        label_nii, rate=mid_num_rate,
        left_ventricle_label=left_ventricle_label, right_ventricle_label=right_ventricle_label,
    )
    mid_image = image_nii.get_slice(mid_num, dim=2)
    mid_label = label_nii.get_slice(mid_num, dim=2)
    quad_seg_point = get_quad_seg_point(
        mid_image, mid_label, rate=quad_seg_rate,
        pons_label=pons_label
    )
    pons_area = get_pons_area(mid_label, pons_label=pons_label)
    midbrain_area = get_midbrain_area(mid_label, midbrain_label=midbrain_label)
    return mid_num, quad_seg_point, midbrain_area, pons_area

def show(image_nii, mid_num, quad_seg_point, point_color=255):
    process.show(
        process.add_points(
            image_nii.get_slice(mid_num, dim=2),
            [quad_seg_point], 1, point_color
        )
    )
