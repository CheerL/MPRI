import cv2
import matplotlib.pyplot as plt
import numpy as np

from process.base_image import ImageProcess

class SecondStageProcess(ImageProcess):
    def get_region_2_mask(self, shape, point, angle=175, height=40, width=50):
        return self.get_rect_mask(shape, point, height, width, angle)

    def get_silces_2(self, nii, mid_num, num=30, dim=0):
        start, end = int(mid_num - num / 2), int(mid_num + num / 2)
        return nii.get_slice(start, end, dim)

    def get_mcp(self, img, region_2_mask, rate=2, test=False):
        region_2_img = img * region_2_mask
        region_2_otsu = self.get_otsu(region_2_img)
        region_2_bin_img = self.get_binary_image(region_2_img, region_2_otsu * rate)
        components, label = self.get_connected_component(region_2_bin_img)

        try:
            mcp = components[0]
        except IndexError:
            mcp = label

        if test:
            test_data = {
                'r1_img': region_2_img,
                'r1_bin_img': region_2_bin_img,
                'r1_otsu': region_2_otsu,
                'label': label,
            }
            return mcp, test_data
        return mcp

    def get_mcp_part_seg_point(self, part, min_dis=2, type_='u', quality=0.001):
        assert type_ in ['u', 'd']
        points = [
            self.get_revese_point(point)
            for point in np.concatenate(
                cv2.goodFeaturesToTrack(
                    part.img_uint8,
                    10, quality, 2
                ).astype(int)
            ) if part.left + min_dis <= point[0] < part.right - min_dis
        ]
        assert points
        if len(points) > 1:
            points = sorted(
                points,
                key=lambda x: x[0],
                reverse=True if type_ == 'u' else False
            )
        return points[0]

    def get_mcp_seg_point(self, mcp, left_rate=1/2, right_rate=1/5,
                          min_dis=2, min_part_area=10, quality=0.001):
        left_bound = int((1-left_rate) * mcp.left + left_rate * mcp.right)
        right_bound = int(right_rate * mcp.left + (1-right_rate) * mcp.right)
        part_img = mcp.canvas
        for y in range(mcp.up, mcp.down):
            for x in range(left_bound, right_bound):
                if not mcp.img[y, x]:
                    part_img[y, x] = 255

        part_img[mcp.up-1, left_bound:right_bound] = 255
        part_img[mcp.down, left_bound:right_bound] = 255
        components, label = self.get_connected_component(part_img, min_area=min_part_area)
        assert len(components) == 2
        up_part, down_part = sorted(components, key=lambda x: x.centroid[0])
        up_point = self.get_mcp_part_seg_point(up_part, min_dis, 'u', quality)
        down_point = self.get_mcp_part_seg_point(down_part, min_dis, 'd', quality)

        ext_img = mcp.canvas
        for y in range(mcp.up, up_point[0]):
            for x in range(up_point[1], right_bound):
                if mcp.img[y, x]:
                    ext_img[y, x] = 255

        components, _ = self.get_connected_component(ext_img, min_area=5)
        assert components
        ext_part = components[0]
        ext_point = ext_part.get_bound_point('u')
        return up_point, down_point, ext_point

    def get_mcp_slice_length(self, up_point, down_point, ext_point,
                             sup_limit=15, inf_limit=4):
        if inf_limit < abs(up_point[0] - ext_point[0]) < sup_limit:
            return self.get_distance(up_point, down_point)
        raise AttributeError()

    def get_mcp_length(self, slices_2, region_2, rate=2,
                       left_rate=1/2, right_rate=1/5,
                       min_dis=2, min_part_area=10, quality=0.001,
                       sup_limit=15, inf_limit=4):
        mcps = [self.get_mcp(img, region_2, rate) for img in slices_2]
        mcp_slice_length = []
        for mcp in mcps:
            try:
                mcp_points  = self.get_mcp_seg_point(
                    mcp, left_rate, right_rate,
                    min_dis, min_part_area, quality
                )
                mcp_length = self.get_mcp_slice_length(*mcp_points, sup_limit, inf_limit)
                mcp_slice_length.append(mcp_length)
            except:
                pass
        return np.mean(mcp_slice_length)