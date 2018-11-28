import cv2
import matplotlib.pyplot as plt
import numpy as np

from process.base_image import ImageProcess

class FirstStageProcess(ImageProcess):
    def get_slices_1(self, nii, num=10, dim=0):
        return self.get_mid_silces(nii, num, dim)

    def get_center_component(self, label, info, rate=1/4):
        height, width = label.shape
        up_limit, left_limit = rate * height, rate * width
        down_limit, right_limit = height - up_limit, width - left_limit
        return [
            (level, stat, centroid)
            for level, stat, centroid in info
            if (
                stat[0] > left_limit and
                stat[1] > up_limit and
                stat[0] + stat[2] < right_limit and
                stat[1] + stat[3] < down_limit
            )
        ]

    def get_near_component(self, label, info, pos, limit, type_='point'):
        if type_ == 'point':
            near_component = [
                (level, stat, centroid)
                for level, stat, centroid in info
                if self.get_area_distance(label == level, pos) < limit
            ]
        elif type_ == 'area':
            near_component = [
                (level, stat, centroid)
                for level, stat, centroid in info
                if self.get_area_distance(pos, centroid) < limit
            ]
        else:
            raise TypeError()
        return near_component

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
        label, info = self.get_connected_component(clahe_bin_img, min_area)
        info = self.get_center_component(label, info, center_rate)
        info = self.get_near_component(label, info, center, max_distance)
        try:
            corpus_info = info[0]
            corpus = label == corpus_info[0]
        except IndexError:
            corpus_info = info
            corpus = label

        if test:
            test_data = {
                'bin_img': bin_img,
                'clahe_img': clahe_img,
                'clahe_bin_img': clahe_bin_img,
                'label': label,
                'center': center,
                'info': info
            }
            return corpus, corpus_info, test_data
        return corpus, corpus_info

    # def get_nii_cc(self, silces,
    #                clahe_limit=0.02, clahe_row=8, clahe_col=8,
    #                clahe_bin_add=0.2, center_rate=1/4,
    #                min_area=200, max_distance=45):
    #                #clahe_bin_add=0.15 for MSA
    #     return [
    #         self.get_cc(
    #             img, clahe_limit, clahe_row, clahe_col,
    #             clahe_bin_add, center_rate,
    #             min_area, max_distance
    #         ) for img in silces
    #     ]

    def get_smallest_component(self, component_list):
        sorted_component_list = sorted(
            enumerate(component_list),
            key=lambda x:x[1][1][1][4]
            )
        num, component = sorted_component_list[0]
        return component, num

    def get_component_bound(self, component, info, type_='lr'):
        if type_ == 'lr':
            y_pos, x_pos = np.where(component)
            left_x, right_x = int(info[1][0]), int(info[1][0] + info[1][2] - 1)
            left_y = int(y_pos[np.where(x_pos == left_x)[0]].mean())
            right_y = int(y_pos[np.where(x_pos == right_x)[0]].mean())
            return (left_y, left_x), (right_y, right_x)
        elif type_ == 'ud':
            y_pos, x_pos = np.where(component)
            up_y, down_y = int(info[1][1]), int(info[1][1] + info[1][3] - 1)
            up_x = int(x_pos[np.where(y_pos == up_y)[0]].mean())
            down_x = int(x_pos[np.where(y_pos == down_y)[0]].mean())
            return (up_y, up_x), (down_y, down_x)
        elif type_ == 'all':
            left_bound, right_bound = self.get_component_bound(component, info, 'lr')
            up_bound, down_bound = self.get_component_bound(component, info, 'ud')
            return left_bound, right_bound, up_bound, down_bound

    def get_region_1_mask(self, component, info, rate=2/3, upside=True):
        left_bound, right_bound = self.get_component_bound(component, info, 'lr')
        width = int(self.get_distance(left_bound, right_bound))
        height = int(width * rate)
        diff_y, diff_x = right_bound[0] - left_bound[0], right_bound[1] - left_bound[1]
        angle = -np.rad2deg(np.arctan(diff_y / diff_x))
        return self.get_region_mask(
            component.shape,
            right_bound if upside else left_bound,
            height, width,
            180 + angle if upside else angle
            )

    def get_upper_brainstem(self, img, region_1_mask, rate=2.2, test=False):
        region_1_img = img * region_1_mask
        region_1_otsu = self.get_otsu(region_1_img)
        region_1_bin_img = self.get_binary_image(region_1_img, region_1_otsu * rate)
        label, info = self.get_connected_component(region_1_bin_img)

        try:
            upper_brainstem_info = info[0]
            upper_brainstem = label == upper_brainstem_info[0]
        except IndexError:
            upper_brainstem_info = info
            upper_brainstem = label

        if test:
            test_data = {
                'r1_img': region_1_img,
                'r1_bin_img': region_1_bin_img,
                'r1_otsu': region_1_otsu,
                'label': label,
                'info': info
            }
            return upper_brainstem, upper_brainstem_info, test_data
        return upper_brainstem, upper_brainstem_info

    def get_midsagittal(self, upper_brainstem_list, slices_1):
        upper_brainstem, num = self.get_smallest_component(upper_brainstem_list)
        return upper_brainstem, slices_1[num]

    def get_brainstem(self, img, upper_brainstem,
                      clahe_limit=0.02, clahe_row=8, clahe_col=8,
                      clahe_bin_add=0.15, min_area=500, max_distance=5,
                      test=False):
        center = self.get_grav_center(upper_brainstem.astype(np.uint8))
        clahe_img = self.get_clahe_image(img, clahe_limit, clahe_row, clahe_col)
        clahe_otsu = self.get_otsu(clahe_img)
        clahe_bin_img = self.get_binary_image(clahe_img, clahe_otsu + 255 * clahe_bin_add)
        label, info = self.get_connected_component(clahe_bin_img, min_area)
        info = self.get_near_component(label, info, center, max_distance)
        try:
            brainstem_info = info[0]
            brainstem = label == brainstem_info[0]
        except IndexError:
            brainstem_info = info
            brainstem = label

        if test:
            test_data = {
                'clahe_img': clahe_img,
                'clahe_bin_img': clahe_bin_img,
                'clahe_otsu': clahe_otsu,
                'label': label,
                'center': center,
                'info': info
            }
            return brainstem, brainstem_info, test_data
        return brainstem, brainstem_info

    def get_quadrigeminal(self, img, brainstem, region_1_mask,
                          clahe_limit=0.02, clahe_row=8, clahe_col=8,
                          clahe_bin_add=0.15, min_area=25, max_distance=5,
                          test=False):
        # center = self.get_grav_center(upper_brainstem.astype(np.uint8))
        clahe_img = self.get_clahe_image(img, clahe_limit, clahe_row, clahe_col)
        clahe_otsu = self.get_otsu(clahe_img)
        clahe_bin_img = self.get_binary_image(clahe_img, clahe_otsu + 255 * clahe_bin_add)
        clahe_bin_img = clahe_bin_img * ~brainstem * region_1_mask

        label, info = self.get_connected_component(clahe_bin_img, min_area)
        info = self.get_near_component(label, info, brainstem, max_distance, type_='area')
        try:
            quadrigeminal_info = info[0]
            quadrigeminal = label == quadrigeminal_info[0]
        except IndexError:
            quadrigeminal_info = info
            quadrigeminal = label

        if test:
            test_data = {
                'clahe_img': clahe_img,
                'clahe_bin_img': clahe_bin_img,
                'clahe_otsu': clahe_otsu,
                'label': label,
                'info': info
            }
            return quadrigeminal, quadrigeminal_info, test_data
        return quadrigeminal, quadrigeminal_info

    def get_brainstem_seg_point(self, brainstem, brainstem_info,
                                size=3, max_num=5, quality=0.5, min_dis=10,
                                min_dis_from_bound=5):
        img = brainstem.astype(np.uint8) * 255
        kernel = np.ones((size, size))
        img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
        img = self.get_side_contour(img, 'r')
        corners = cv2.goodFeaturesToTrack(img, max_num, quality, min_dis).astype(np.int)
        corners = [self.get_revese_point(point) for point in np.concatenate(corners)]
        brainstem_size_info = brainstem_info[1]
        left, up = brainstem_size_info[0], brainstem_size_info[1]
        right, down = left + brainstem_size_info[2] - 1, up + brainstem_size_info[3] - 1

        corners = [
            point for point in corners
            if (
                left + min_dis_from_bound < point[1] < right - min_dis_from_bound and
                up + min_dis_from_bound < point[0] < down - min_dis_from_bound
            )
        ]
        assert len(corners) == 2
        return sorted(corners, key=lambda x: x[0])


class SecendStageProcess(ImageProcess):
    def get_region_2_mask(self, shape, point, angle=160, height=50, width=70):
        return self.get_region_mask(shape, point, height, width, angle)

    def get_silces_2(self, nii, num=30, dim=0):
        return self.get_mid_silces(nii, num, dim)
