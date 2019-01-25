import numpy as np
import process
import component

def get_mid_num(
    label_nii, dim=2, rate=0.2,
    left_ventricle_label=8, right_ventricle_label=9
):
    size = label_nii.size[dim]
    min_volum = 500
    for num, label in enumerate(label_nii.get_slice(
        start=int((1-rate)*size/2),
        end=int((1+rate)*size/2),
        dim=dim
    )):
        volum = np.logical_or(
            label==left_ventricle_label,
            label==right_ventricle_label
        ).sum()
        if volum < min_volum:
            min_volum = volum
            mid_num = int((1-rate)*size/2) + num

    return mid_num

def get_quad_seg_point(mid_image, mid_label, label_type=25, bin_rate=1.2):
    box = ((-20, 10), (-10, 10))
    right_bound = process.get_bound_point(mid_label==label_type, 'r')
    box_index = [
        slice(right_bound[0]+box[0][0], right_bound[0]+box[0][1]),
        slice(right_bound[1]+box[1][0], right_bound[1]+box[1][1])
    ]
    otsu = process.get_otsu(mid_image[box_index])
    bin_image = process.get_binary_image(mid_image, otsu*bin_rate)
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

def get_brainstem_seg_point(mid_label, quad_seg_point, label_types=[25, 26]):
    box = ((-20, 20), (-50, -10))
    for y in range(quad_seg_point[0]+box[0][0], quad_seg_point[0]+box[0][1]):
        for x in range(quad_seg_point[1]+box[1][0], quad_seg_point[1]+box[1][1]):
            block = mid_label[y-1:y+2, x-1:x+2]
            if not mid_label[y, x] and label_types[0] in block and label_types[1] in block:
                brainstem_seg_point = (y, x)
    return brainstem_seg_point
