import numpy as np

import component
import process


def _get_mcp_up_point(mcp, up, left, right):
    for y in range(up, up+50):
        if np.all(mcp.img[y, left:right]):
            # print(mcp.img[y, left:right], )
            break
    false_list = mcp.img[y-1, left:right]
    x = np.where(false_list == False)[0].mean()
    return y-1, left+int(x)

def _get_mcp_down_point(mcp, right):
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
    label, medulla_label=27, scp_label=28,
    max_medulla_vol=100, max_scp_vol=0
):
    medulla = label == medulla_label
    scp = label == scp_label
    return medulla.sum() <= max_medulla_vol and scp.sum() <= max_scp_vol

def get_mcp_seg_point(
    image, label, quad_seg_point,
    rate=0.8, midbrain_label=26, box=((10, 35), (-8, 8))
):
    midbrain = label == midbrain_label
    midbrain_pos = np.where(midbrain)
    midbrain_mean = image[midbrain_pos].mean()
    midbrain_right = midbrain_pos[1].max()
    up = quad_seg_point[0]+box[0][0]
    down = quad_seg_point[0]+box[0][1]
    left = midbrain_right+box[1][0]
    right = midbrain_right+box[1][1]

    bin_image = process.clear_bottleneck(image > (midbrain_mean * rate))
    mask = np.zeros(image.shape, bool)
    mask[up:down, left:right] = True
    mcp = component.get_connected_component((bin_image*mask).astype(np.uint8))[0][0]
    down_point = _get_mcp_down_point(mcp, right)
    up_point = _get_mcp_up_point(mcp, up, left, right)
    return up_point, down_point, mcp

def get_mcp_width(up_point, down_point):
    return process.get_distance(up_point, down_point)

def run(
    image_nii, label_nii, quad_seg_point, mid_num,
    midbrain_label=26, medulla_label=27, scp_label=28,
    max_medulla_vol=100, max_scp_vol=0, num=13, rate=0.8,
    box=((10, 35), (-8, 8)), show=False
):
    mcp_widths = []
    show_info = []
    for index, (image, label) in enumerate(zip(
        image_nii.get_slice(mid_num-num, mid_num+num, dim=2),
        label_nii.get_slice(mid_num-num, mid_num+num, dim=2)
    )):
        if is_mcp_slice(label):
            up_point, down_point, mcp = get_mcp_seg_point(
                image, label, quad_seg_point,
                rate=rate, midbrain_label=midbrain_label, box=box
            )
            mcp_widths.append(get_mcp_width(up_point, down_point))
            if show:
                show_info.append((mid_num-num+index, (up_point, down_point, mcp)))

    mcp_mean_width = np.mean(mcp_widths)
    return mcp_mean_width, show_info

def show(image_nii, show_info, mask_color=25, point_color=255):
    process.show([
        process.add_points(
            process.add_mask(
                image_nii.get_slice(index, dim=2),
                mcp.img_bool, mask_color
            ), (up, down), 1, point_color
        ) for index, (up, down, mcp) in show_info
    ])
