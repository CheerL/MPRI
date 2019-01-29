from typing import List, Tuple

import numpy as np
import SimpleITK as sitk

from config import Mcp_show_info_item, Point, Scp_show_info_item


def calc(pons_area: int, midbrain_area: int, mcp_width: float, scp_width: float) -> None:
    mrpi = (pons_area / midbrain_area) * (mcp_width / scp_width)
    print('Pons area: {}, Midbrain area: {}, MCP width: {}, SCP width: {}, MRPI: {}'. format(
        pons_area, midbrain_area, mcp_width, scp_width, mrpi
    ))

def save_to_sitk(
    file_name: str, size: Tuple[int, int], mid_num: int, quad_seg_point: Point,
    mcp_show_info: List[Mcp_show_info_item], scp_show_info: List[Scp_show_info_item]
) -> None:
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
