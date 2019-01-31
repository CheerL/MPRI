from typing import List, Tuple

import numpy as np
from itertools import product
import component
import process
from config import Box, Mcp_seg_result, Mcp_show_info_item, Point
from DM.file_manage import LabelNiiFileManager, RotatedNiiFileManager


def _get_mcp_up_point(mcp: component.ConnectedComponent) -> Point:
    up = mcp.up
    left = mcp.left
    right = mcp.right
    for y in range(up, up+50):
        if np.all(mcp.img[y, left:right]):
            # print(mcp.img[y, left:right], )
            break
    false_list = mcp.img[y-1, left:right]
    x = np.where(false_list == False)[0].mean()
    return y-1, left+int(x)

def _get_mcp_down_point(mcp: component.ConnectedComponent, up_point: Point, max_right: int=7) -> Point:
    up, left = up_point
    right = min(left + max_right, mcp.right)
    down = mcp.down
    # print(down-up, right-left)
    distance = [
        ((y, x), process.get_distance((y, x), up_point))
        for y, x in product(range(up+1, down+1), range(left, right+1))
        if not mcp.img[y, x]
    ]
    distance.sort(key=lambda x: x[1])
    return distance[0][0]
    # allow_right = True
    # allow_left = True

    # y_count = 0
    # x_count = 0
    # max_y_count = 3
    # max_x_count = 3
    # right = mcp.right
    # left = mcp.left
    # up = mcp.up
    # op_list = []

    # y, _ = mcp.get_bound_point('d')
    # pos = np.where(mcp.img[y])[0]
    # if pos.max() < right:
    #     x = pos.max() + 1
    # elif pos.min() > left:
    #     x = pos.min() - 1
    # else:
    #     raise IndexError('No start point')

    # try:
    #     while True:
    #         # if y_count < max_y_count and not mcp.img[y-1, x]:
    #         #     y -= 1
    #         #     y_count += 1
    #         # elif mcp.img[y-1, x]:
    #         #     y_count = 0
    #         #     x_right = np.where(mcp.img[y-1, x:]==False)[0].min() + x
    #         #     x_left = np.where(mcp.img[y-1, :x]==False)[0].max()
    #         #     if x_right < right and x_right - x < max_x_count:
    #         #         if not mcp.img[y, x:x_right+1]:
    #         #             x = x_right
    #         #             y -= 1
    #         #             continue
    #         #     if 
    #         # op_list.append((y, x))
    #         # if mcp.img[y, x] and x < right and allow_right:
    #         #     x += 1
    #         # elif mcp.img[y, x] and x >= right:
    #         #     allow_right = False
    #         #     x -= 1
    #         # elif mcp.img[y, x] and not allow_right:
    #         #     x -= 1
    #         # else:
    #         #     if not mcp.img[y-1, x] and y > up:
    #         #         y -= 1
    #         #         y_count += 1
    #         #         x_count = 0
    #         #     elif y <= up:
    #         #         raise IndexError('Come to the top')
    #         #     else:
    #         #         if y_count >= max_y_count:
    #         #             if mcp.img[y, x-1]:
    #         #                 return y, x
    #         #             elif allow_left:
    #         #                 x -= 1
    #         #                 x_count += 1
    #         #             elif not allow_left
    #         #         else:
    #         #             if allow_right:
    #         #                 y_count = 0
    #         #             if x >= right or x_count >= max_x_count:
    #         #                 allow_right = False
    #         #                 x_count = 0
    #         #             if allow_right and not mcp.img[y, x+1]:
    #         #                 x += 1
    #         #                 x_count += 1
    #         #             elif not allow_right and not mcp.img[y, x-1]:
    #         #                 x -= 1
    #         #             else:
    #         #                 return y, x
    # except:
    #     process.show([
    #         mcp.img_uint8[120:160, 120:150],
    #         process.add_points(
    #             mcp.img_uint8,
    #             op_list[:50],
    #             0, 127
    #         )[120:160, 120:150]
    #     ])
    #     raise EnvironmentError()

def is_mcp_slice(
    label: np.ndarray, medulla_label: int=27, scp_label: int=28,
    max_medulla_vol: int=100, max_scp_vol: int=0
) -> bool:
    medulla = label == medulla_label
    scp = label == scp_label
    return medulla.sum() <= max_medulla_vol and scp.sum() <= max_scp_vol

def get_mcp_seg_point(
    image: np.ndarray, label: np.ndarray, quad_seg_point: Point,
    rate: float=0.8, pons_label: int=26, box: Box=((10, 40), (-8, 8))
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
    up_point = _get_mcp_up_point(mcp)
    down_point = _get_mcp_down_point(mcp, up_point)
    return up_point, down_point, mcp

def get_mcp_width(up_point: Point, down_point: Point) -> float:
    return process.get_distance(up_point, down_point)

def run(
    image_nii: RotatedNiiFileManager, label_nii: LabelNiiFileManager, quad_seg_point: Point, mid_num: int,
    pons_label: int=26, medulla_label: int=27, scp_label: int=28, num: int=14, rate: float=0.82,
    max_medulla_vol: int=150, max_scp_vol: int=0, min_dis: int=3, max_dis: int=13,
    box: Box=((10, 40), (-8, 8)), show: bool=False
) -> Tuple[int, List[Mcp_show_info_item]]:
    mcp_widths = []
    show_info = []
    for index, (image, label) in enumerate(zip(
        image_nii.get_slice(mid_num-num, mid_num+num, dim=2),
        label_nii.get_slice(mid_num-num, mid_num+num, dim=2)
    )):
        if is_mcp_slice(
            label, medulla_label=medulla_label, scp_label=scp_label,
            max_medulla_vol=max_medulla_vol, max_scp_vol=max_scp_vol
        ):
            try:
                up_point, down_point, mcp = get_mcp_seg_point(
                    image, label, quad_seg_point,
                    rate=rate, pons_label=pons_label, box=box
                )
            except IndexError as error:
                print(error)
                continue

            left_part = mcp.img_bool.copy()
            left_part[:, down_point[1]:] = False
            mcp_down = np.where(left_part)[0].max()
            # print(index, mcp_down-down_point[0], mcp.down - down_point[0])
            if min_dis <= mcp_down - down_point[0] <= max_dis and mcp.down - down_point[0] < 20:
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
