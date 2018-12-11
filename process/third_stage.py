import cv2
import numpy as np
import functools
import config
import process
from component import ConnectedComponent
from process.base_stage import (BaseStageProcess, TooBigException,
                                TooLessException, TooManyException,
                                TooSmallExceotion)


class ThirdStageProcess(BaseStageProcess):
    def __init__(self):
        self.mask = None
        self.scp_components = None
        self.scp_slice_num = None
        self.scp_slice = None
        self.scp_peduncles = None
        self.scp_lengths = None
        super().__init__()

    def init_para(self):
        return {
            'get_slices': {
                'point_dim': 1,
                'num': 10,
                'dim': 1
            },
            'get_region_mask': {
                'height': 25,
                'width': 20
            },
            'get_scp_components': {
                'rate': 1.14,
                'min_area': 10
            },
            'get_scp_slice_num': {
                'reverse': True
            },
            'get_scp_peduncle': {
                'clahe_limit': 0.03,
                'clahe_bin_add': 0.18,
                'min_area': 10,
                'max_dis': 10,
                'clahe_col': config.CALHE_COL,
                'clahe_row': config.CALHE_ROW,
            },
            'get_scp_width': {
                'max_width': 10,
                'min_width': 2
            }
        }

    def run(self, nii, stage):
        def scp_peduncles_run():
            self.scp_peduncles = self.get_scp_peduncle(self.scp_slice, stage.quad_point)
            self.scp_lengths = [
                self.get_scp_width(peduncle) for peduncle in self.scp_peduncles
            ]

        def scp_peduncles_exception(error):
            if isinstance(error, (TooManyException, TooSmallExceotion)):
                self.para['get_scp_peduncle']['clahe_bin_add'] -= 0.01
            elif isinstance(error, (TooLessException, TooBigException)):
                self.para['get_scp_peduncle']['clahe_bin_add'] += 0.01
            else:
                raise error

        self.nii = nii
        self.slices = self.get_slices(stage.quad_point)
        self.slice_normalize()
        self.mask = self.get_region_mask(stage.quad_point, stage.real_mid_num)
        self.scp_components = [
            self.get_scp_components(img) for img in self.slices
        ]
        self.scp_slice_num = self.get_scp_slice_num()
        self.scp_slice = self.slices[self.scp_slice_num]
        self.error_retry(scp_peduncles_run, scp_peduncles_exception)

    def show(self):
        process.show([
            self.scp_slice,
            self.scp_slice * self.mask,
            process.add_mask(self.scp_slice, functools.reduce(
                lambda a, b: b + a, self.scp_components[self.scp_slice_num]
            )),
            process.add_mask(self.scp_slice, self.scp_peduncles[0] + self.scp_peduncles[1])
        ])

    def get_slices(self, point, point_dim, num, dim):
        mid_num = point[point_dim]
        start, end = int(mid_num - num / 2), int(mid_num + num / 2)
        return self.nii.get_slice(start, end, dim)

    def get_region_mask(self, point, real_mid_num, height, width):
        start_point = (point[0]-int(height/2), real_mid_num-int(width/2))
        return process.get_rect_mask(self.slices[0].shape, start_point, height, width)

    def get_scp_components(self, img, rate, min_area, test=False):
        masked_img = img * self.mask
        otsu = process.get_otsu(process.get_limit_img(masked_img))
        bin_img = process.get_binary_image(masked_img, otsu*rate)
        components, label = ConnectedComponent.get_connected_component(bin_img, min_area)
        if test:
            test_data = {
                'masked_img': masked_img,
                'otsu': otsu,
                'bin_img': bin_img,
                'label': label
            }
            return components, test_data
        return components

    def get_scp_slice_num(self, reverse):
        components = reversed(self.scp_components) if reverse else self.scp_components
        for num, component in enumerate(components):
            if len(component) == 3:
                return len(self.slices) - num - 1 if reverse else num

    def get_scp_peduncle(self, img, point, clahe_limit, clahe_bin_add,
                         clahe_row, clahe_col, min_area, max_dis, test=False):
        clahe_img = process.get_clahe_image(img, clahe_limit, clahe_row, clahe_col)
        otsu = process.get_otsu(clahe_img)
        bin_img = process.get_binary_image(clahe_img, otsu + 255 * clahe_bin_add)

        components, label = ConnectedComponent.get_connected_component(bin_img, min_area)
        components = process.get_near_component(components, point, max_dis)
        components = [
            component for component in components
            if not point in component
        ]
        if len(components) > 2:
            raise TooManyException('Peduncles are too many.')
        elif len(components) < 2:
            raise TooLessException('Peduncles are too less.')

        if test:
            test_data = {
                'clahe_img': clahe_img,
                'bin_img': bin_img,
                'otsu': otsu,
                'label': label
            }
            return components, test_data
        return components

    def get_scp_width(self, component, min_width, max_width, test=False):
        contours = cv2.findContours(
            component.img_uint8,
            cv2.RETR_TREE,
            cv2.CHAIN_APPROX_NONE
            )
        min_rect = cv2.minAreaRect(contours[1][0])
        # box = np.int32(np.around(cv2.boxPoints(min_rect)))
        box = cv2.boxPoints(min_rect)
        width = min(
            process.get_distance(box[0], box[1]),
            process.get_distance(box[0], box[2]),
            process.get_distance(box[0], box[3])
        )
        if width > max_width:
            raise TooBigException('SCP is to big')
        elif width < min_width:
            raise TooSmallExceotion('SCP is to small')

        if test:
            test_data = {
                'box': np.around(box).astype(np.int),
            }
            return width, test_data
        return width
