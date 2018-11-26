import cv2
import matplotlib.pyplot as plt
import numpy as np

from process.base_image import ImageProcess

class FirstStageProcess(ImageProcess):
    def get_slices_1(self, nii, num, dim=0):
        return nii.get_slice(
            int((nii.size[dim]-num)/2),
            int((nii.size[dim]+num)/2),
            dim
        )

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

    def get_near_component(self, label, info, point, limit):
        return [
            (level, stat, centroid)
            for level, stat, centroid in info
            if self.get_area_distance(label == level, point) < limit
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

    def get_component_bound(self, component, info):
        y_pos, x_pos = np.where(component)
        left_x, right_x = int(info[1][0]), int(info[1][0] + info[1][2] - 1)
        left_y = int(y_pos[np.where(x_pos == left_x)[0]].mean())
        right_y = int(y_pos[np.where(x_pos == right_x)[0]].mean())
        return (left_y, left_x), (right_y, right_x)

    def get_region_1(self, component, info, angle_negative=False, rate=2/3):
        left_bound, right_bound = self.get_component_bound(component, info)
        angle = self.get_vert_angle(left_bound, right_bound, angle_negative)
        distance = self.get_distance(left_bound, right_bound)
        vert_distance = distance * rate
        left_vert_bound = self.get_point_by_angle_and_distance(left_bound, angle, vert_distance)
        right_vert_bound = self.get_point_by_angle_and_distance(right_bound, angle, vert_distance)
        return left_bound, right_bound, right_vert_bound, left_vert_bound

    def get_region_1_mask(self, img, corpus, info, angle_negative=False, rate=2/3, reverse=True):
        region_1 = self.get_region_1(corpus, info, angle_negative, rate)
        if reverse:
            region_1 = [self.get_revese_point(point) for point in region_1]
        canvas = np.zeros(img.shape)
        return cv2.fillPoly(canvas, [np.array(region_1)], 255).astype(bool)

    def get_upper_brainstem(self, img, region_1_mask, rate=2.2, test=False):
        region_1_img = img.copy()
        region_1_img[~region_1_mask] = 0
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

    def get_brainstem(self, img, upper_brainstem,
                      clahe_limit=0.02, clahe_row=8, clahe_col=8,
                      clahe_bin_add=0.2, min_area=200, max_distance=1,
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
