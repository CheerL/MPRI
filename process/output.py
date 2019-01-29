import numpy as np
import SimpleITK as sitk

def save_to_sitk(
    file_name, size, mid_num, quad_seg_point,
    mcp_show_info, scp_show_info
):
    result = np.zeros(size, dtype=np.uint8)
    # quad_seg_point
    result[quad_seg_point][mid_num] = 1
    # mcp and mcp_seg_point
    for index, (up_point, down_point, mcp) in mcp_show_info:
        result[:, :, index][mcp.img_bool] = 2
        result[up_point][index] = 3
        result[down_point][index] = 3
    for index, scps in scp_show_info:
        for scp in scps:
            result[:, index, :][scp.img_bool] = 4

    result = np.rot90(result, k=2, axes=(0, 1))
    result = sitk.GetImageFromArray(result)
    sitk.WriteImage(result, file_name)

def save_to_image():
    raise NotImplementedError()