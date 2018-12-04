import cv2
import numpy as np
import config
from process import base_process
from component import ConnectedComponent

class TooLessException(Exception):
    pass

class TooManyException(Exception):
    pass

class FirstStageProcess(object):
    def __init__(self):
        self.nii = None
        self.slices = None
        self.corpuses = None
        self.masks = None
        self.upper_brainstems = None
        self.mid_num = None
        self.mid_img = None
        self.mid_mask = None
        self.mid_corpus = None
        self.mid_upper_brainstem = None
        self.brainstem = None
        self.quad = None
        self.up_brainstem_point = None
        self.down_brainstem_point = None
        self.quad_point = None
        self.para = self.get_para()

    def run(self, nii):
        self.nii = nii
        self.slices = self.get_slices()
        self.corpuses = [
            self.get_corpus(img) for img in self.slices
        ]
        self.masks = [
            self.get_region_mask(corpus) for corpus in self.corpuses
        ]
        self.upper_brainstems = [
            self.get_upper_brainstem(img, mask)
            for img, mask in zip(self.slices, self.masks)
        ]
        self.mid_num = self.get_mid_num()
        self.mid_img = self.slices[self.mid_num]
        self.mid_mask = self.masks[self.mid_num]
        self.mid_corpus = self.corpuses[self.mid_num]
        self.mid_upper_brainstem = self.upper_brainstems[self.mid_num]
        self.brainstem = self.get_brainstem()
        self.quad = self.get_quad()
        self.down_brainstem_point, self.up_brainstem_point = self.get_brainstem_seg_point()
        self.quad_point = self.get_quad_seg_point()

    def get_para(self):
        return {
            'get_slices': {
                'num': config.S1_num,
                'dim': config.S1_dim
            },
        }

    @base_process.default_para
    def get_slices(self, num, dim):
        dim_size = self.nii.size[dim]
        start, end = int((dim_size - num) / 2), int((dim_size + num) / 2)
        return self.nii.get_slice(start, end, dim)

    def get_region_mask(self, component, rate=2/3, upside=True):
        left_bound, right_bound = component.get_bound_point('lr')
        width = int(base_process.get_distance(left_bound, right_bound))
        height = int(width * rate)
        angle = base_process.get_angle(left_bound, right_bound)
        return base_process.get_rect_mask(
            component.shape,
            (right_bound if upside else left_bound),
            height, width,
            (180 - angle if upside else -angle)
            )

    def get_corpus(self, img, clahe_limit=0.02, clahe_bin_add=0.2, center_rate=1/4,
                   min_area=200, max_distance=45,
                   clahe_row=config.CALHE_ROW, clahe_col=config.CALHE_COL,
                   test=False):
                   #clahe_bin_add=0.15 for MSA
        _, bin_img = base_process.get_otsu(img, False)
        center = base_process.get_grav_center(bin_img)
        clahe_img = base_process.get_clahe_image(img, clahe_limit, clahe_row, clahe_col)
        clahe_otsu = base_process.get_otsu(clahe_img)
        clahe_bin_img = base_process.get_binary_image(clahe_img, clahe_otsu + 255 * clahe_bin_add)
        components, label = ConnectedComponent.get_connected_component(clahe_bin_img, min_area)
        components = [
            component for component in components
            if component.in_range(
                left=img.shape[1] * center_rate,
                up=img.shape[0] * center_rate,
                right=img.shape[1] * (1 - center_rate),
                down=img.shape[0] * (1 - center_rate)
            )
        ]
        components = base_process.get_near_component(components, center, max_distance)
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

    def get_upper_brainstem(self, img, mask, rate=2.2, test=False):
        masked_img = img * mask
        otsu = base_process.get_otsu(masked_img)
        bin_img = base_process.get_binary_image(masked_img, otsu * rate)
        components, label = ConnectedComponent.get_connected_component(bin_img)
        upper_brainstem = components[0]

        if test:
            test_data = {
                'masked_img': masked_img,
                'bin_img': bin_img,
                'otsu': otsu,
                'label': label,
            }
            return upper_brainstem, test_data
        return upper_brainstem

    def get_mid_num(self):
        numed_upper_brainstems = sorted(
            enumerate(self.upper_brainstems),
            key=lambda x:x[1].area
            )
        return numed_upper_brainstems[0][0]

    def get_real_midsagittal_num(self, nii, mid_num, slice_num, dim):
        return int((nii.size[dim] - slice_num) / 2) + mid_num

    def get_brainstem(self, clahe_limit=0.02, clahe_row=8, clahe_col=8,
                      clahe_bin_add=0.15, min_area=500, max_distance=5,
                      test=False):
        center = base_process.get_grav_center(self.mid_upper_brainstem.img_uint8)
        clahe_img = base_process.get_clahe_image(self.mid_img, clahe_limit, clahe_row, clahe_col)
        clahe_otsu = base_process.get_otsu(clahe_img)
        clahe_bin_img = base_process.get_binary_image(clahe_img, clahe_otsu + 255 * clahe_bin_add)
        components, label = ConnectedComponent.get_connected_component(clahe_bin_img, min_area)
        components = base_process.get_near_component(components, center, max_distance)
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

    def get_quad(self, clahe_limit=0.02, clahe_row=8, clahe_col=8,
                 clahe_bin_add=0.15, min_area=25, max_distance=5,
                 test=False):
        clahe_img = base_process.get_clahe_image(self.mid_img, clahe_limit, clahe_row, clahe_col)
        clahe_otsu = base_process.get_otsu(clahe_img)
        clahe_bin_img = base_process.get_binary_image(clahe_img, clahe_otsu + 255 * clahe_bin_add)
        clahe_bin_img = clahe_bin_img * ~self.brainstem.img * self.mid_mask

        components, label = ConnectedComponent.get_connected_component(clahe_bin_img, min_area)
        components = base_process.get_near_component(
            components, self.brainstem.img, max_distance, type_='area'
        )
        quadrigeminal = components[0]

        if test:
            test_data = {
                'clahe_img': clahe_img,
                'clahe_bin_img': clahe_bin_img,
                'clahe_otsu': clahe_otsu,
                'label': label,
            }
            return quadrigeminal, test_data
        return quadrigeminal

    def get_brainstem_seg_point(self, close_kernel_size=3, max_num=5, quality=0.5,
                                corner_min_dis=10, min_dis=5):
        img = self.brainstem.img_uint8.copy()
        img = cv2.morphologyEx(
            img, cv2.MORPH_CLOSE,
            np.ones((close_kernel_size, close_kernel_size))
        )
        img = base_process.get_side_contour(img, 'r')
        corners = cv2.goodFeaturesToTrack(img, max_num, quality, corner_min_dis).astype(np.int)
        corners = [base_process.get_revese_point(point) for point in np.concatenate(corners)]

        corners = [
            point for point in corners
            if (
                self.brainstem.left + min_dis < point[1] < self.brainstem.right - min_dis and
                self.brainstem.up + min_dis < point[0] < self.brainstem.down - min_dis
            )
        ]
        if len(corners) < 2:
            raise TooLessException('Too less points found when segement brainstem')
        elif len(corners) > 2:
            raise TooManyException('Too many points found when segement brainstem')
        return sorted(corners, key=lambda x: x[0])

    def get_quad_seg_point(self):
        return self.quad.get_bound_point('u')
