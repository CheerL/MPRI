import cv2
import matplotlib.pyplot as plt
import numpy as np

from process.base_image import ImageProcess

class FirstStageProcess(ImageProcess):
    def get_slices_1(self, nii, num=10, dim=0):
        dim_size = nii.size[dim]
        start, end = int((dim_size - num) / 2), int((dim_size + num) / 2)
        return nii.get_slice(start, end, dim)

    def get_center_component(self, components, rate=1/4):
        return [
            component
            for component in components
            if (
                component.left > component.shape[1] * rate and
                component.up > component.shape[0] * rate and
                component.right < component.shape[1] * (1 - rate) and
                component.down < component.shape[0] * (1 - rate)
            )
        ]

    def get_corpus(self, img, clahe_limit=0.02, clahe_row=8, clahe_col=8,
                   clahe_bin_add=0.2, center_rate=1/4,
                   min_area=200, max_distance=45,
                   test=False):
                   #clahe_bin_add=0.15 for MSA
        _, bin_img = self.get_otsu(img, False)
        center = self.get_grav_center(bin_img)
        clahe_img = self.get_clahe_image(img, clahe_limit, clahe_row, clahe_col)
        clahe_otsu = self.get_otsu(clahe_img)
        clahe_bin_img = self.get_binary_image(clahe_img, clahe_otsu + 255 * clahe_bin_add)
        components, label = self.get_connected_component(clahe_bin_img, min_area)
        components = self.get_center_component(components, center_rate)
        components = self.get_near_component(components, center, max_distance)
        try:
            corpus = components[0]
        except IndexError:
            corpus = label

        if test:
            test_data = {
                'bin_img': bin_img,
                'clahe_img': clahe_img,
                'clahe_bin_img': clahe_bin_img,
                'label': label,
                'center': center,
            }
            return corpus, test_data
        return corpus

    def get_region_1_mask(self, component, rate=2/3, upside=True):
        left_bound, right_bound = component.get_bound_point('lr')
        width = int(self.get_distance(left_bound, right_bound))
        height = int(width * rate)
        diff_y, diff_x = right_bound[0] - left_bound[0], right_bound[1] - left_bound[1]
        angle = -np.rad2deg(np.arctan(diff_y / diff_x))
        return self.get_rect_mask(
            component.shape,
            right_bound if upside else left_bound,
            height, width,
            180 + angle if upside else angle
            )

    def get_upper_brainstem(self, img, region_1_mask, rate=2.2, test=False):
        region_1_img = img * region_1_mask
        region_1_otsu = self.get_otsu(region_1_img)
        region_1_bin_img = self.get_binary_image(region_1_img, region_1_otsu * rate)
        components, label = self.get_connected_component(region_1_bin_img)

        try:
            upper_brainstem = components[0]
        except IndexError:
            upper_brainstem = label

        if test:
            test_data = {
                'r1_img': region_1_img,
                'r1_bin_img': region_1_bin_img,
                'r1_otsu': region_1_otsu,
                'label': label,
            }
            return upper_brainstem, test_data
        return upper_brainstem

    def get_midsagittal(self, upper_brainstems):
        numed_upper_brainstems = sorted(
            enumerate(upper_brainstems),
            key=lambda x:x[1].area
            )
        mid_num, upper_brainstem = numed_upper_brainstems[0]
        return upper_brainstem, mid_num

    def get_real_midsagittal_num(self, nii, mid_num, slice_num, dim):
        return int((nii.size[dim] - slice_num) / 2) + mid_num

    def get_brainstem(self, img, upper_brainstem,
                      clahe_limit=0.02, clahe_row=8, clahe_col=8,
                      clahe_bin_add=0.15, min_area=500, max_distance=5,
                      test=False):
        center = self.get_grav_center(upper_brainstem.img.astype(np.uint8))
        clahe_img = self.get_clahe_image(img, clahe_limit, clahe_row, clahe_col)
        clahe_otsu = self.get_otsu(clahe_img)
        clahe_bin_img = self.get_binary_image(clahe_img, clahe_otsu + 255 * clahe_bin_add)
        components, label = self.get_connected_component(clahe_bin_img, min_area)
        components = self.get_near_component(components, center, max_distance)
        try:
            brainstem = components[0]
        except IndexError:
            brainstem = label

        if test:
            test_data = {
                'clahe_img': clahe_img,
                'clahe_bin_img': clahe_bin_img,
                'clahe_otsu': clahe_otsu,
                'label': label,
                'center': center,
            }
            return brainstem, test_data
        return brainstem

    def get_quadrigeminal(self, img, brainstem, region_1_mask,
                          clahe_limit=0.02, clahe_row=8, clahe_col=8,
                          clahe_bin_add=0.15, min_area=25, max_distance=5,
                          test=False):
        # center = self.get_grav_center(upper_brainstem.astype(np.uint8))
        clahe_img = self.get_clahe_image(img, clahe_limit, clahe_row, clahe_col)
        clahe_otsu = self.get_otsu(clahe_img)
        clahe_bin_img = self.get_binary_image(clahe_img, clahe_otsu + 255 * clahe_bin_add)
        clahe_bin_img = clahe_bin_img * ~brainstem.img * region_1_mask

        components, label = self.get_connected_component(clahe_bin_img, min_area)
        components = self.get_near_component(components, brainstem.img, max_distance, type_='area')
        try:
            quadrigeminal = components[0]
        except IndexError:
            quadrigeminal = label

        if test:
            test_data = {
                'clahe_img': clahe_img,
                'clahe_bin_img': clahe_bin_img,
                'clahe_otsu': clahe_otsu,
                'label': label,
            }
            return quadrigeminal, test_data
        return quadrigeminal

    def get_brainstem_seg_point(self, brainstem, size=3,
                                max_num=5, quality=0.5, corner_min_dis=10,
                                min_dis=5):
        img = brainstem.img_uint8.copy()
        kernel = np.ones((size, size))
        img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
        img = self.get_side_contour(img, 'r')
        corners = cv2.goodFeaturesToTrack(img, max_num, quality, corner_min_dis).astype(np.int)
        corners = [self.get_revese_point(point) for point in np.concatenate(corners)]

        corners = [
            point for point in corners
            if (
                brainstem.left + min_dis < point[1] < brainstem.right - min_dis and
                brainstem.up + min_dis < point[0] < brainstem.down - min_dis
            )
        ]
        assert len(corners) == 2
        return sorted(corners, key=lambda x: x[0])


class SecendStageProcess(ImageProcess):
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


class ThirdStageProcess(ImageProcess):
    def get_silces_3(self, nii, point, num, dim=0):
        mid_num = point[1]
        start, end = int(mid_num - num / 2), int(mid_num + num / 2)
        return nii.get_slice(start, end, dim)

    def get_region_3_mask(self, slice_3, point, real_mid_num,
                          height=25, width=20):
        start_point = (point[0]-int(height/2), real_mid_num-int(width/2))
        return self.get_rect_mask(slice_3[0].shape, start_point, height, width)

    def get_region_component(self, img, region_3, rate=0.14, min_area=10, test=False):
        region_img = img * region_3
        otsu = self.get_otsu(img)
        bin_img = self.get_binary_image(region_img, otsu+rate*255)
        components, label = self.get_connected_component(bin_img, min_area)
        if test:
            test_data = {
                'masked_img': region_img,
                'otsu': otsu,
                'bin_img': bin_img,
                'label': label
            }
            return components, test_data
        return components

    def get_scp_slice_num(self, components, slice_3, reverse=True):
        for num, component in enumerate(reversed(components)) if reverse else components:
            if len(component) == 3:
                return len(slice_3) - num - 1 if reverse else num

    def get_scp_peduncle(self, img, point,
                         clahe_limit=0.03, clahe_row=8, clahe_col=8,
                         clahe_bin_add=0.28, min_area=10, max_distance=10,
                         test=False):
        clahe_img = self.get_clahe_image(img, clahe_limit, clahe_row, clahe_col)
        clahe_otsu = self.get_otsu(clahe_img)
        clahe_bin_img = self.get_binary_image(clahe_img, clahe_otsu + 255 * clahe_bin_add)

        components, label = self.get_connected_component(clahe_bin_img, min_area)
        components = self.get_near_component(components, point, max_distance)
        components = [
            component for component in components
            if not component.in_range(point)
        ]
        if test:
            test_data = {
                'clahe_img': clahe_img,
                'clahe_bin_img': clahe_bin_img,
                'clahe_otsu': clahe_otsu,
                'label': label
            }
            return components, test_data
        return components

    def get_scp_width(self, component):
        contours = cv2.findContours(
            component.img_uint8,
            cv2.RETR_TREE,
            cv2.CHAIN_APPROX_NONE
            )
        min_rect = cv2.minAreaRect(contours[1][0])
        # box = np.int32(np.around(cv2.boxPoints(min_rect)))
        box = cv2.boxPoints(min_rect)
        return min(
            self.get_distance(box[0], box[1]),
            self.get_distance(box[0], box[2]),
            self.get_distance(box[0], box[3])
        )
