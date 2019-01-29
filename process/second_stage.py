from typing import List, Tuple

import numpy as np

import component
import process
from config import Box, Mcp_seg_result, Mcp_show_info_item, Point
from DM.file_manage import LabelNiiFileManager, RotatedNiiFileManager


def _get_mcp_up_point(
    mcp: component.ConnectedComponent,
    up: int, left: int, right: int
) -> Point:
    for y in range(up, up+50):
        if np.all(mcp.img[y, left:right]):
            # print(mcp.img[y, left:right], )
            break
    false_list = mcp.img[y-1, left:right]
    x = np.where(false_list == False)[0].mean()
    return y-1, left+int(x)

def _get_mcp_down_point(mcp: component.ConnectedComponent, right: int) -> Point:
    allow_right = True
    y, x = mcp.get_bound_point('d')
    y_count = 0
    x_count = 0
    max_y_count = 3
    max_x_count = 3
    while True:
        if mcp.img[y, x]:
            x += 1
        else:
            if not mcp.img[y-1, x]:
                y -= 1
                y_count += 1
                x_count = 0
            elif y_count > max_y_count:
                if mcp.img[y, x-1]:
                    return y, x
                else:
                    x -= 1
            else:
                y_count = 0
                if x == right - 1 or x_count == max_x_count:
                    allow_right = False
                if allow_right and not mcp.img[y, x+1]:
                    x += 1
                    x_count += 1
                elif not mcp.img[y, x-1]:
                    x -= 1
                    allow_right = False
                else:
                    return y, x

def is_mcp_slice(
    label: np.ndarray, medulla_label: int=27, scp_label: int=28,
    max_medulla_vol: int=100, max_scp_vol: int=0
) -> bool:
    medulla = label == medulla_label
    scp = label == scp_label
    return medulla.sum() <= max_medulla_vol and scp.sum() <= max_scp_vol

def get_mcp_seg_point(
    image: np.ndarray, label: np.ndarray, quad_seg_point: Point,
    rate: float=0.8, pons_label: int=26, box: Box=((10, 35), (-8, 8))
) -> Mcp_seg_result:
    pons = label == pons_label
    pons_pos = np.where(pons)
    pons_mean = image[pons_pos].mean()
    pons_right = pons_pos[1].max()
    up = quad_seg_point[0]+box[0][0]
    down = quad_seg_point[0]+box[0][1]
    left = pons_right+box[1][0]
    right = pons_right+box[1][1]

    bin_image = process.clear_bottleneck(image > (pons_mean * rate))
    mask = np.zeros(image.shape, bool)
    mask[up:down, left:right] = True
    mcp = component.get_connected_component((bin_image*mask).astype(np.uint8))[0][0]
    down_point = _get_mcp_down_point(mcp, right)
    up_point = _get_mcp_up_point(mcp, up, left, right)
    return up_point, down_point, mcp

def get_mcp_width(up_point: Point, down_point: Point) -> float:
    return process.get_distance(up_point, down_point)

def run(
    image_nii: RotatedNiiFileManager, label_nii: LabelNiiFileManager, quad_seg_point: Point, mid_num: int,
    pons_label: int=26, medulla_label: int=27, scp_label: int=28,
    max_medulla_vol: int=100, max_scp_vol: int=0, num: int=13, rate: float=0.8,
    box: Box=((10, 35), (-8, 8)), show: bool=False
) -> Tuple[int, List[Mcp_show_info_item]]:
    mcp_widths = []
    show_info = []
    for index, (image, label) in enumerate(zip(
        image_nii.get_slice(mid_num-num, mid_num+num, dim=2),
        label_nii.get_slice(mid_num-num, mid_num+num, dim=2)
    )):
        if is_mcp_slice(label):
            up_point, down_point, mcp = get_mcp_seg_point(
                image, label, quad_seg_point,
                rate=rate, pons_label=pons_label, box=box
            )
            mcp_widths.append(get_mcp_width(up_point, down_point))
            if show:
                show_info.append((mid_num-num+index, (up_point, down_point, mcp)))

    mcp_mean_width = np.mean(mcp_widths)
    return mcp_mean_width, show_info

def show(
    image_nii: RotatedNiiFileManager, show_info: List[Mcp_show_info_item],
    mask_color: int=25, point_color: int=255
) -> None:
    process.show([
        process.add_points(
            process.add_mask(
                image_nii.get_slice(index, dim=2),
                mcp.img_bool, mask_color
            ), (up, down), 1, point_color
        ) for index, (up, down, mcp) in show_info
    ])
