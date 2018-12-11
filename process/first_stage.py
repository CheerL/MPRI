import cv2
import numpy as np

import config
import process
from component import ConnectedComponent
from process.base_stage import (BaseStageProcess, TooLessException,
                                TooManyException, TooSmallExceotion)


class FirstStageProcess(BaseStageProcess):
    def __init__(self):
        self.corpuses = None
        self.masks = None
        self.filter_nums = None
        self.upper_brainstems = None
        self.mid_num = None
        self.mid_img = None
        self.mid_mask = None
        self.mid_corpus = None
        self.mid_upper_brainstem = None
        self.real_mid_num = None
        self.brainstem = None
        self.quad = None
        self.up_brainstem_point = None
        self.down_brainstem_point = None
        self.quad_point = None
        super().__init__()

    def run(self, nii, stage=None):
        def corpus_run():
            self.corpuses = [
                self.get_corpus(img) for img in self.slices
            ]

        def corpus_exception(error):
            if isinstance(error, IndexError):
                self.para['get_corpus']['bin_rate'] += 0.02
            else:
                raise error

        def corpus_out_exception(error):
            self.corpuses = []
            self.para['get_corpus']['bin_rate'] = config.CORPUS_RATE
            for index, img in enumerate(self.slices):
                try:
                    corpus = self.get_corpus(img)
                except:
                    corpus = self.corpuses[index-1] if index is not 0 else None
                finally:
                    self.corpuses.append(corpus)

            num = len(self.corpuses)
            for index, corpus in enumerate(self.corpuses[::-1]):
                if corpus is None:
                    self.corpuses[num - index -1] = self.corpuses[num - index]

            if None in self.corpuses:
                raise error

        def brainstem_run():
            self.brainstem = self.get_brainstem()

        def brainstem_exception(error):
            if isinstance(error, (IndexError, TooSmallExceotion)):
                self.para['get_brainstem']['bin_rate'] -= 0.01
            else:
                raise error

        def quad_run():
            self.quad = self.get_quad()

        def quad_exception(error):
            if isinstance(error, IndexError):
                self.para['get_quad']['bin_rate'] -= 0.01

        def brainstem_seg_run():
            self.down_brainstem_point, self.up_brainstem_point = self.get_brainstem_seg_point()

        def brainstem_seg_exception(error):
            if isinstance(error, TooManyException):
                self.para['get_brainstem_seg_point']['quality'] += 0.02
            elif isinstance(error, TooLessException):
                self.para['get_brainstem_seg_point']['quality'] -= 0.02
            else:
                raise error

        self.nii = nii
        self.slices = self.get_slices()
        self.slice_normalize()
        try:
            self.error_retry(corpus_run, corpus_exception)
        except RuntimeError as error:
            corpus_out_exception(error)

        masks_and_paras = list(zip(*[
            self.get_region_mask(corpus) for corpus in self.corpuses
        ]))
        self.masks = masks_and_paras[0]
        self.filter_nums = self.get_filter_nums(masks_and_paras)
        self.upper_brainstems = [
            (self.get_upper_brainstem(img, mask)
             if index in self.filter_nums
             else ConnectedComponent.get_connected_component(
                 np.ones(mask.shape, np.uint8))[0][0])
            for index, (img, mask) in enumerate(zip(self.slices, self.masks))
        ]
        self.mid_num = self.get_mid_num()
        self.mid_img = self.slices[self.mid_num]
        self.mid_mask = self.masks[self.mid_num]
        self.mid_corpus = self.corpuses[self.mid_num]
        self.mid_upper_brainstem = self.upper_brainstems[self.mid_num]
        self.error_retry(brainstem_run, brainstem_exception, 10)
        self.error_retry(quad_run, quad_exception, 10)
        self.error_retry(brainstem_seg_run, brainstem_seg_exception, 10)
        self.quad_point = self.get_quad_seg_point()
        self.real_mid_num = self.get_real_mid_num()

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
                'rate': 5/6
            },
            'get_corpus': {
                'clahe_limit': 0.02,
                'bin_rate': config.CORPUS_RATE,
                'center_rate': 1/4,
                'min_area': 200,
                'max_dis': 45,
                'clahe_col': config.CALHE_COL,
                'clahe_row': config.CALHE_ROW,
            },
            'get_upper_brainstem': {
                'bin_rate': config.BRAINSTEM_RATE
            },
            'get_brainstem': {
                'clahe_limit': 0.02,
                'bin_rate': config.BRAINSTEM_RATE,
                'min_area': 500,
                'max_dis': 5,
                'clahe_col': config.CALHE_COL,
                'clahe_row': config.CALHE_ROW,
                'diff': 5
            },
            'get_quad': {
                'clahe_limit': 0.02,
                'bin_rate': 1.5,
                'min_area': 15,
                'max_area': 55,
                'max_dis': 8,
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

    @staticmethod
    def get_filter_nums(masks_and_paras):
        corpuses, left_bounds, right_bounds, angles = masks_and_paras
        left_bounds = np.array(left_bounds)
        right_bounds = np.array(right_bounds)
        angles = np.array(angles)
        filter_nums = list(range(len(corpuses)))
        while True:
            mean_left = np.mean(left_bounds[filter_nums], axis=0)
            left_dis = [
                process.get_distance(mean_left, point)
                for point in left_bounds[filter_nums]
                ]
            if max(left_dis) > 15:
                del filter_nums[np.argmax(left_dis)]
            else:
                break

        while True:
            mean_right = np.mean(right_bounds[filter_nums], axis=0)
            right_dis = [
                process.get_distance(mean_right, point)
                for point in right_bounds[filter_nums]
                ]
            if max(right_dis) > 15:
                del filter_nums[np.argmax(right_dis)]
            else:
                break

        while True:
            mean_angle = np.mean(angles[filter_nums], axis=0)
            angle_diff = [
                abs(mean_angle - angle)
                for angle in angles[filter_nums]
                ]
            if max(angle_diff) > 10:
                del filter_nums[np.argmax(angle_diff)]
            else:
                break

        return filter_nums

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
            ), left_bound, right_bound, angle

    def get_real_mid_num(self):
        dim = self.para['get_slices']['dim']
        num = self.para['get_slices']['num']
        return int((self.nii.size[dim] - num) / 2) + self.mid_num

    def get_corpus(self, img, clahe_limit, bin_rate, center_rate,
                   min_area, max_dis, clahe_row, clahe_col, test=False):
        _, bin_img = process.get_otsu(img, False)
        center = process.get_grav_center(bin_img)
        clahe_img = process.get_clahe_image(img, clahe_limit, clahe_row, clahe_col)
        clahe_otsu = process.get_otsu(clahe_img)
        clahe_bin_img = process.get_binary_image(clahe_img, clahe_otsu * bin_rate)
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
        components = process.get_near_component(components, center, max_dis)
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

    def get_upper_brainstem(self, img, mask, bin_rate, test=False):
        masked_img = img * mask
        otsu = process.get_otsu(process.get_limit_img(masked_img))
        # otsu = process.get_otsu(img)
        bin_img = process.get_binary_image(masked_img, otsu * bin_rate)
        components, label = ConnectedComponent.get_connected_component(bin_img)
        upper_brainstem = components[0]
        upper_brainstem.info = {'otsu': otsu}

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

    def get_brainstem(self, clahe_limit, bin_rate, clahe_row, clahe_col,
                      min_area, max_dis, diff, test=False):
        center = self.mid_upper_brainstem.centroid
        # clahe_img = process.get_clahe_image(self.mid_img, clahe_limit, clahe_row, clahe_col)
        # clahe_otsu = process.get_otsu(clahe_img)
        otsu = self.mid_upper_brainstem.info['otsu']
        clahe_bin_img = process.get_binary_image(self.mid_img, otsu * bin_rate)
        components, label = ConnectedComponent.get_connected_component(clahe_bin_img, min_area)
        components = process.get_near_component(components, center, max_dis)
        brainstem = components[0]
        if self.mid_upper_brainstem.down - brainstem.down > diff:
            raise TooSmallExceotion('Brainstem is too small.')

        if test:
            test_data = {
                # 'clahe_img': clahe_img,
                'clahe_bin_img': clahe_bin_img,
                'otsu': otsu,
                'label': label,
                'center': center,
            }
            return brainstem, test_data
        return brainstem

    def get_quad(self, clahe_limit, bin_rate, clahe_row, clahe_col,
                 min_area, max_area, max_dis, test=False):
        clahe_img = process.get_clahe_image(self.mid_img, clahe_limit, clahe_row, clahe_col)
        clahe_otsu = process.get_otsu(clahe_img)
        clahe_bin_img = process.get_binary_image(clahe_img, clahe_otsu * bin_rate)
        clahe_bin_img = clahe_bin_img * ~(self.brainstem.img_bool + self.mid_upper_brainstem.img_bool) * self.mid_mask
        components, label = ConnectedComponent.get_connected_component(clahe_bin_img, min_area)
        components = [component for component in components if component.area <= max_area]
        components = process.get_near_component(
            components, self.mid_upper_brainstem.img, max_dis, type_='area'
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
        bound_y, bound_x = self.mid_upper_brainstem.get_bound_point('u')
        img[:bound_y, :bound_x] = 255
        process.show(img)
        corners = cv2.goodFeaturesToTrack(
            img, max_corner_num, quality, min_corner_dis
            ).astype(np.int)
        corners = [process.get_revese_point(point) for point in np.concatenate(corners)]
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
