import cv2
import numpy as np

import config
import process
from component import ConnectedComponent
from process.base_stage import BaseStageProcess


class TooLessException(Exception):
    pass

class TooManyException(Exception):
    pass

class NotEnoughException(Exception):
    pass

class FirstStageProcess(BaseStageProcess):
    def __init__(self):
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
        super().__init__()

    def run(self, nii, stage=None):
        def brainstem_run():
            self.brainstem = self.get_brainstem()

        def brainstem_exception(error):
            if isinstance(error, (IndexError, NotEnoughException)):
                self.para['get_brainstem']['clahe_bin_add'] -= 0.01
            else:
                raise error

        def quad_run():
            self.quad = self.get_quad()

        def quad_exception(error):
            if isinstance(error, IndexError):
                self.para['get_quad']['clahe_bin_add'] -= 0.01

        def brainstem_seg_run():
            self.down_brainstem_point, self.up_brainstem_point = self.get_brainstem_seg_point()

        def brainstem_seg_exception(error):
            if isinstance(error, TooManyException):
                self.para['get_brainstem_seg_point']['quality'] += 0.1
            elif isinstance(error, TooLessException):
                self.para['get_brainstem_seg_point']['quality'] -= 0.1
            else:
                raise error

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
        self.error_retry(brainstem_run, brainstem_exception)
        self.error_retry(quad_run, quad_exception)
        self.error_retry(brainstem_seg_run, brainstem_seg_exception)
        self.quad_point = self.get_quad_seg_point()

    def show(self):
        process.show([
            self.mid_img,
            self.mid_img * self.mid_mask,
            process.add_mask(
                self.mid_img,
                self.mid_corpus.img_bool +
                self.brainstem.img_bool
            ),
            process.add_mask(
                self.mid_img,
                self.mid_upper_brainstem.img_bool
            ),
            process.add_mask(
                self.mid_img,
                self.quad.img_bool
            ),
            process.add_points(
                self.mid_img,
                [
                    self.up_brainstem_point,
                    self.down_brainstem_point,
                    self.quad_point
                ],
                size=2,
            )
        ])

    def init_para(self):
        return {
            'get_slices': {
                'num': config.S1_num,
                'dim': config.S1_dim
            },
            'get_region_mask': {
                'rate': 2/3
            },
            'get_corpus': {
                'clahe_limit': 0.02,
                'clahe_bin_add': 0.2, #0.15 for MSA
                'center_rate': 1/4,
                'min_area': 200,
                'max_distance': 45,
                'clahe_col': config.CALHE_COL,
                'clahe_row': config.CALHE_ROW,
            },
            'get_upper_brainstem': {
                'rate': 1.3
            },
            'get_brainstem': {
                'clahe_limit': 0.02,
                'clahe_bin_add': 0.15,
                'min_area': 500,
                'max_distance': 5,
                'clahe_col': config.CALHE_COL,
                'clahe_row': config.CALHE_ROW,
                'diff': 5
            },
            'get_quad': {
                'clahe_limit': 0.02,
                'clahe_bin_add': 0.15,
                'min_area': 15,
                'max_area': 50,
                'max_distance': 8,
                'clahe_col': config.CALHE_COL,
                'clahe_row': config.CALHE_ROW,
            },
            'get_brainstem_seg_point': {
                'close_kernel_size': 3,
                'max_corner_num': 5,
                'quality': 0.3,
                'min_corner_dis': 10,
                'min_dis': 5
            }
        }

    def get_slices(self, num, dim):
        dim_size = self.nii.size[dim]
        start, end = int((dim_size - num) / 2), int((dim_size + num) / 2)
        return self.nii.get_slice(start, end, dim)

    def get_region_mask(self, component, rate, upside=True):
        left_bound, right_bound = component.get_bound_point('lr')
        width = int(process.get_distance(left_bound, right_bound))
        height = int(width * rate)
        angle = process.get_angle(left_bound, right_bound)
        return process.get_rect_mask(
            component.shape,
            (right_bound if upside else left_bound),
            height, width,
            (180 - angle if upside else -angle)
            )

    def get_corpus(self, img, clahe_limit, clahe_bin_add, center_rate,
                   min_area, max_distance, clahe_row, clahe_col, test=False):
        _, bin_img = process.get_otsu(img, False)
        center = process.get_grav_center(bin_img)
        clahe_img = process.get_clahe_image(img, clahe_limit, clahe_row, clahe_col)
        clahe_otsu = process.get_otsu(clahe_img)
        clahe_bin_img = process.get_binary_image(clahe_img, clahe_otsu + 255 * clahe_bin_add)
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
        components = process.get_near_component(components, center, max_distance)
        corpus = components[0]

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

    def get_upper_brainstem(self, img, mask, rate, test=False):
        masked_img = img * mask
        otsu = process.get_otsu(process.get_limit_img(masked_img))
        bin_img = process.get_binary_image(masked_img, otsu * rate)
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
            key=lambda x: x[1].area
            )
        return numed_upper_brainstems[0][0]

    def get_brainstem(self, clahe_limit, clahe_bin_add, clahe_row, clahe_col,
                      min_area, max_distance, diff, test=False):
        center = process.get_grav_center(self.mid_upper_brainstem.img_uint8)
        clahe_img = process.get_clahe_image(self.mid_img, clahe_limit, clahe_row, clahe_col)
        clahe_otsu = process.get_otsu(clahe_img)
        clahe_bin_img = process.get_binary_image(clahe_img, clahe_otsu + 255 * clahe_bin_add)
        components, label = ConnectedComponent.get_connected_component(clahe_bin_img, min_area)
        components = process.get_near_component(components, center, max_distance)
        brainstem = components[0]
        if self.mid_upper_brainstem.down - brainstem.down > diff:
            raise NotEnoughException

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

    def get_quad(self, clahe_limit, clahe_bin_add, clahe_row, clahe_col,
                 min_area, max_area, max_distance, test=False):
        clahe_img = process.get_clahe_image(self.mid_img, clahe_limit, clahe_row, clahe_col)
        clahe_otsu = process.get_otsu(clahe_img)
        clahe_bin_img = process.get_binary_image(clahe_img, clahe_otsu + 255 * clahe_bin_add)
        clahe_bin_img = clahe_bin_img * ~self.brainstem.img * self.mid_mask

        components, label = ConnectedComponent.get_connected_component(clahe_bin_img, min_area)
        components = [component for component in components if component.area < max_area]
        components = process.get_near_component(
            components, self.mid_upper_brainstem.img, max_distance, type_='area'
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

    def get_brainstem_seg_point(self, close_kernel_size, max_corner_num,
                                quality, min_corner_dis, min_dis):
        img = self.brainstem.img_uint8.copy()
        img = cv2.morphologyEx(
            img, cv2.MORPH_CLOSE,
            np.ones((close_kernel_size, close_kernel_size))
        )
        # process.show(img)
        img = process.get_side_contour(img, 'r')
        corners = cv2.goodFeaturesToTrack(
            img, max_corner_num, quality, min_corner_dis
            ).astype(np.int)
        corners = [process.get_revese_point(point) for point in np.concatenate(corners)]
        # print(corners)
        # process.show(process.add_points(img, corners, size=2, color=127))
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
