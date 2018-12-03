import cv2
import matplotlib.pyplot as plt
import numpy as np

from process.base_image import ImageProcess

class TooLessException(Exception):
    pass

class TooManyException(Exception):
    pass

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
            (right_bound if upside else left_bound),
            height, width,
            (180 + angle if upside else angle)
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
        brainstem = components[0]

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
        # self.show(img)
        corners = cv2.goodFeaturesToTrack(img, max_num, quality, corner_min_dis).astype(np.int)
        corners = [self.get_revese_point(point) for point in np.concatenate(corners)]

        corners = [
            point for point in corners
            if (
                brainstem.left + min_dis < point[1] < brainstem.right - min_dis and
                brainstem.up + min_dis < point[0] < brainstem.down - min_dis
            )
        ]
        if len(corners) < 2:
            raise TooLessException('Too less points found when segement brainstem')
        elif len(corners) > 2:
            raise TooManyException('Too many points found when segement brainstem')
        return sorted(corners, key=lambda x: x[0])
