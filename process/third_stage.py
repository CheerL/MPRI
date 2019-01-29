import cv2
import numpy as np

import component
import process


def get_scp_sum(label_nii, quad_seg_point, scp_label=28):
    quad_num = int(quad_seg_point[1])

    for num in range(quad_num-3, quad_num+5):
        label = label_nii.get_slice(num, dim=1)
        scp = label==scp_label
        if scp.sum():
            return num-1

def get_scp_components(
    image, label, quad_point,
    scp_label=28, midbrain_label=26, rate=0.95, box=((2, 13), (-10, 10))
):
    mask_box = (
        slice(quad_point[0]+box[0][0], quad_point[0]+box[0][1]),
        slice(quad_point[1]+box[1][0], quad_point[1]+box[1][1])
    )
    mask = np.zeros(image.shape, bool)
    mask[mask_box] = True
    scp = np.logical_or(label==scp_label, label==midbrain_label)
    scp_mean = image[np.where(scp*mask)].mean()
    bin_scp = process.clear_bottleneck(image * mask > scp_mean * rate)
    components = component.get_connected_component(bin_scp.astype(np.uint8), 10)[0]
    big_component = components[0]
    if big_component.right - big_component.left > 16:
        for index, row in enumerate(big_component.img[big_component.down-1:big_component.up:-1]):
            pos = np.where(row)[0]
            true_row = row[pos.min():(pos.max()+1)]
            if not all(true_row):
                stop_index = index
                false_pos = np.where(true_row == False)[0] + pos.min()
                break
            
        big_component.img[
            big_component.down-stop_index-1:big_component.down,
            [false_pos]
        ] = 0
        scp_components, _ = component.get_connected_component(
            big_component.img_uint8, 5
        )
    else:
        scp_components = components[:2]
    assert len(scp_components) == 2
    return scp_components

def get_scp_width(component, min_width=2, max_width=10):
    contours = cv2.findContours(
        component.img_uint8,
        cv2.RETR_TREE,
        cv2.CHAIN_APPROX_NONE
        )
    min_rect = cv2.minAreaRect(contours[1][0])
    # box = np.int32(np.around(cv2.boxPoints(min_rect)))
    box = cv2.boxPoints(min_rect)
    width = min(
        process.get_distance(box[0], box[1]),
        process.get_distance(box[0], box[2]),
        process.get_distance(box[0], box[3])
    )
    assert min_width < width < max_width
    # if test:
    #     test_data = {
    #         'box': np.around(box).astype(np.int),
    #     }
    #     return width, test_data
    return width

def run(
    image_nii, label_nii, quad_seg_point, mid_num,
    num=2, scp_label=28, midbrain_label=26, rate=0.95,
    max_width=10, min_width=2, box=((2, 13), (-10, 10)), show=False
):
    scp_num = get_scp_sum(label_nii, quad_seg_point, scp_label=scp_label)
    quad_point = (quad_seg_point[0], mid_num)
    scp_widths = []
    show_info = []

    for index, (image, label) in enumerate(zip(
        image_nii.get_slice(scp_num, scp_num+num, dim=1),
        label_nii.get_slice(scp_num, scp_num+num, dim=1)
    )):
        scp_components = get_scp_components(
            image, label, quad_point,
            scp_label=scp_label, midbrain_label=midbrain_label, rate=rate
        )
        if show:
            show_info.append((scp_num+index, scp_components))
        for scp_component in scp_components:
           scp_widths.append(get_scp_width(
               scp_component,
               min_width=min_width, max_width=max_width
            ))
    scp_mean_width = np.mean(scp_widths)
    return scp_mean_width, show_info

def show(image_nii, show_info, mask_color=25):
    process.show([
        process.add_mask(
            process.add_mask(
                image_nii.get_slice(index, dim=1), components[0].img_bool, mask_color
            ), components[1].img_bool, mask_color
        ) for index, components in show_info
    ])
