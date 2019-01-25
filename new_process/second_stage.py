import numpy as np
import component

def clear_bottleneck(image):
    height, width = image.shape
    for y in range(1,height-1):
        for x in range(1,width-1):
            if image[y, x]:
                if not image[y-1, x] and not image[y+1, x]:
                    image[y, x] = False
                elif not image[y, x-1] and not image[y, x+1]:
                    image[y, x] = False
    return image

def get_up_point(mcp, up, left, right):
    for y in range(up, up+50):
        if np.all(mcp.img[y, left:right]):
            # print(mcp.img[y, left:right], )
            break
    false_list = mcp.img[y-1, left:right]
    x = np.where(false_list == False)[0].mean()
    return y-1, left+int(x)

def get_down_point(mcp, right):
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

def get_seg_point(
    image, label, brainstem_seg_point, rate=0.8,
    pons_label=26, medulla_label=27, scp_label=28,
    max_medulla_vol=100, max_scp_vol=0, box=((5, 30), (-8, 7))
):
    medulla = label == medulla_label
    scp = label == scp_label
    if medulla.sum() <= max_medulla_vol and scp.sum() <= max_scp_vol:
        pons = label == pons_label
        pons_pos = np.where(pons)
        pons_mean = image[pons_pos].mean()
        pons_right = pons_pos[1].max()
        up = brainstem_seg_point[0]+box[0][0]
        down = brainstem_seg_point[0]+box[0][1]
        left = pons_right+box[1][0]
        right = pons_right+box[1][1]

        bin_image = clear_bottleneck(image > (pons_mean * rate))
        mask = np.zeros(image.shape, bool)
        mask[up:down, left:right] = True
        mcp = component.get_connected_component((bin_image*mask).astype(np.uint8))[0][0]
        down_point = get_down_point(mcp, right)
        up_point = get_up_point(mcp, up, left, right)
        return up_point, down_point, mcp
    else:
        return None
