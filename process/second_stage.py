import cv2
import numpy as np

import process
from component import ConnectedComponent
from process.base_stage import BaseStageProcess


class SecondStageProcess(BaseStageProcess):
    def __init__(self):
        self.angle = None
        self.mask = None
        self.mcps = None
        self.mcp_seg_points = None
        self.mcp_seg_num = None
        self.mcp_length = None
        self.mcp_mean_length = None
        super().__init__()

    def init_para(self):
        return {
            'get_slices': {
                'num': 30,
                'dim': 2
            },
            'get_region_mask': {
                'height': 40,
                'width': 50
            },
            'get_mcp': {
                'rate': 1.25
            },
            'get_mcp_part': {
                'left_rate': 1/2,
                'right_rate': 1/5,
                'min_area': 10
            },
            'get_mcp_part_seg_point': {
                'min_dis': 0,
                'quality': 0.001
            },
            'get_mcp_ext_seg_point': {
                'min_area': 5,
                'right_rate': 1/5,
                'min_dis': 1
            },
            'get_mcp_slice_length': {
                'sup_limit': 15,
                'inf_limit': 3
            }
        }

    def run(self, nii, stage):
        self.nii = nii
        self.angle = self.get_rotate_angle(stage)
        self.slices = self.get_slices(stage)
        self.mask = self.get_region_mask(
            stage.mid_img.shape, stage.up_brainstem_point
        )
        self.mcps = [
            self.get_mcp(img) for img in self.slices
        ]
        self.mcp_seg_points = []
        self.mcp_seg_num = []
        for num, mcp in enumerate(self.mcps):
            try:
                points = self.get_mcp_seg_point(mcp)
                self.mcp_seg_points.append(points)
                self.mcp_seg_num.append(num)
            except AssertionError:
                pass

        self.mcp_length = [
            self.get_mcp_slice_length(*points) for points in self.mcp_seg_points
        ]
        self.mcp_mean_length = np.mean([length for length in self.mcp_length if length > 0])

    def show(self):
        process.show([
            process.add_points(
                process.add_mask(self.slices[num], self.mcps[num].img_bool),
                points, size=3, color=128
                )
            for num, points in zip(self.mcp_seg_num, self.mcp_seg_points)
        ])

    def get_slices(self, stage, num, dim):
        start, end = int(stage.real_mid_num - num / 2), int(stage.real_mid_num + num / 2)
        slices = self.nii.get_slice(start, end, dim)
        rot_matrix = cv2.getRotationMatrix2D(
            process.get_revese_point(stage.up_brainstem_point),
            self.angle, 1
        )
        rot_shape = (stage.mid_img.shape[1], stage.mid_img.shape[0])
        return [
            cv2.warpAffine(
                img,
                rot_matrix,
                rot_shape,
                flags=cv2.INTER_LINEAR
            ) for img in slices
        ]

    def get_rotate_angle(self, stage):
        point_1, point_2 = stage.mid_corpus.get_bound_point('lr')
        return process.get_angle(point_1, point_2)

    def get_region_mask(self, shape, point, height, width):
        return process.get_rect_mask(shape, point, height, width, 180)

    def get_mcp(self, img, rate, test=False):
        masked_img = img * self.mask
        otsu = process.get_otsu(process.get_limit_img(masked_img))
        bin_img = process.get_binary_image(masked_img, otsu * rate)
        components, label = ConnectedComponent.get_connected_component(bin_img)
        mcp = components[0]

        if test:
            test_data = {
                'masked_img': masked_img,
                'bin_img': bin_img,
                'otsu': otsu,
                'label': label,
            }
            return mcp, test_data
        return mcp

    def get_mcp_part(self, mcp, left_rate, right_rate, min_area):
        left_bound = int((1-left_rate) * mcp.left + left_rate * mcp.right)
        right_bound = int(right_rate * mcp.left + (1-right_rate) * mcp.right)
        part_img = mcp.canvas
        for y in range(mcp.up, mcp.down):
            for x in range(left_bound, right_bound):
                if not mcp.img[y, x]:
                    part_img[y, x] = 255
        part_img[mcp.up-1, left_bound:right_bound] = 255
        part_img[mcp.down, left_bound:right_bound] = 255
        # process.show(part_img)
        components, _ = ConnectedComponent.get_connected_component(part_img, min_area)
        assert len(components) == 2
        return sorted(components, key=lambda x: x.centroid[0])

    def get_mcp_part_seg_point(self, component, type_, min_dis, quality):
        assert type_ in ['u', 'd']
        points = [
            process.get_revese_point(point)
            for point in np.concatenate(
                cv2.goodFeaturesToTrack(
                    component.img_uint8,
                    10, quality, 2
                ).astype(int)
            ) if component.left + min_dis <= point[0] < component.right - min_dis
        ]
        assert points
        points = sorted(
            points,
            key=lambda x: x[0],
            reverse=True if type_ == 'u' else False
        )
        return points[0]

    def get_mcp_ext_seg_point(self, mcp, point, right_rate, min_area, min_dis):
        ext_img = mcp.canvas
        right_bound = int(right_rate * mcp.left + (1-right_rate) * mcp.right)
        for y in range(mcp.up, point[0]):
            for x in range(point[1], right_bound):
                if mcp.img[y, x]:
                    ext_img[y, x] = 255

        components, _ = ConnectedComponent.get_connected_component(ext_img, min_area)
        assert components
        ext_part = components[0]
        # process.show(ext_part)
        point = ext_part.get_bound_point('u')
        assert point[1] - ext_part.left >= min_dis
        return point

    def get_mcp_seg_point(self, mcp):
        [up_part, down_part] = self.get_mcp_part(mcp)
        up_point = self.get_mcp_part_seg_point(up_part, type_='u')
        down_point = self.get_mcp_part_seg_point(down_part, type_='d')
        ext_point = self.get_mcp_ext_seg_point(mcp, up_point)
        return up_point, down_point, ext_point

    def get_mcp_slice_length(self, up_point, down_point, ext_point, sup_limit, inf_limit):
        if inf_limit < abs(up_point[0] - ext_point[0]) < sup_limit:
            return process.get_distance(up_point, down_point)
        return 0
