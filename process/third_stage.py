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
        self.quad_point = None
        self.scp_components = None
        self.scp_slice_num = None
        self.scp_slice = None
        self.scp_peduncles = None
        self.scp_lengths = None
        self.scp_mean_length = None
        super().__init__()

    def init_para(self):
        return {
            'get_slices': {
                'point_dim': 1,
                'num': 10,
                'dim': 1
            },
            'get_region_mask': {
                'height': 20,
                'width': 20
            },
            'get_scp_components': {
                'rate': 1.65,
                'min_area': 10
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
        def scp_run():
            self.scp_components = [
                self.get_scp_components(img) for img in self.slices
            ]
            self.scp_slice_num = self.get_scp_slice_num()
            self.scp_slice = self.slices[self.scp_slice_num]
            self.scp_peduncles = self.get_scp_peduncle(self.scp_slice, stage.quad_point)
            self.scp_lengths = [
                self.get_scp_width(peduncle) for peduncle in self.scp_peduncles
            ]

        def scp_exception(error):
            if isinstance(error, (TooManyException, TooSmallExceotion)):
                self.para['get_scp_components']['rate'] -= 0.025
            elif isinstance(error, (TooLessException, TooBigException)):
                self.para['get_scp_components']['rate'] += 0.025
            else:
                raise error

        self.nii = nii
        self.quad_point = stage.quad_point
        self.slices = self.get_slices(stage.quad_point)
        self.slice_normalize()
        self.mask = self.get_region_mask(stage.quad_point, stage.real_mid_num)
        self.error_retry(scp_run, scp_exception)
        self.scp_mean_length = np.mean(self.scp_lengths)

    def show(self):
        process.show([
            self.scp_slice,
            self.scp_slice * self.mask,
            process.add_mask(self.scp_slice, functools.reduce(
                lambda a, b: b + a, self.scp_components[self.scp_slice_num]
            )),
            process.add_mask(self.scp_slice, self.scp_peduncles[0] + self.scp_peduncles[1])
        ])

    @property
    def real_scp_slice_num(self):
        mid_num = self.quad_point[self.para['get_slices']['point_dim']]
        start = int(mid_num - self.para['get_slices']['num'] / 2)
        return start + self.scp_slice_num

    def get_slices(self, point, point_dim, num, dim):
        mid_num = point[point_dim]
        start, end = int(mid_num - num / 2), int(mid_num + num / 2)
        return self.nii.get_slice(start, end, dim)

    def get_region_mask(self, point, real_mid_num, height, width):
        start_point = (point[0]-int(height/1.5), real_mid_num-int(width/2))
        return process.get_rect_mask(self.slices[0].shape, start_point, height, width)

    def get_scp_components(self, img, rate, min_area, test=False):
        masked_img = img * self.mask
        # otsu = process.get_otsu(process.get_limit_img(masked_img))
        otsu = process.get_otsu(img)
        bin_img = process.get_binary_image(masked_img, otsu*rate)
        components, label = ConnectedComponent.get_connected_component(bin_img, min_area)

        # for component in np.array(components):
        #     if component.right - component.left < 5 and (component.)
        if components:
            big_component = sorted(components, key=lambda c: c.area, reverse=True)[0]
            stop_index = 0
            false_pos = []
            if big_component.right - big_component.left > 16:
                for index, row in enumerate(big_component.img[big_component.up:big_component.down]):
                    pos = np.where(row)[0]
                    true_row = row[pos.min():(pos.max()+1)]
                    if not all(true_row):
                        stop_index = index
                        false_pos = np.where(true_row == False)[0] + pos.min()
                        break

                big_component.img[big_component.up:(big_component.up+stop_index), [false_pos]] = 0
                sub_components, label = ConnectedComponent.get_connected_component(
                    big_component.img_uint8, 1
                    )
                components = sub_components + components[1:]

        long_components = [
            component for component in components
            if (component.down - component.up) > self.para['get_region_mask']['height']/1.5
            ]

        for component in long_components:
            img = cv2.erode(component.img_uint8, kernel=np.ones((2, 2)))
            components.append(ConnectedComponent.get_connected_component(img)[0][0])
            index = components.index(component)
            del components[index]


        if test:
            test_data = {
                'masked_img': masked_img,
                'otsu': otsu,
                'bin_img': bin_img,
                'label': label
            }
            return components, test_data
        return components

    def get_scp_slice_num(self):
        numed_components = [
            (index, component) for index, component
            in enumerate(self.scp_components) if len(component) is 3
        ]
        if not numed_components:
            raise TooLessException('No slices have 3 scp components.')

        sorted_numed_components = sorted(
            numed_components, reverse=True,
            key=lambda icc: icc[1][0].area + icc[1][1].area + icc[1][2].area,
        )
        return sorted_numed_components[0][0]

    def get_scp_peduncle(self, img, point, clahe_limit, clahe_bin_add,
                         clahe_row, clahe_col, min_area, max_dis, test=False):
        # clahe_img = process.get_clahe_image(img, clahe_limit, clahe_row, clahe_col)
        # otsu = process.get_otsu(clahe_img)
        # bin_img = process.get_binary_image(clahe_img, otsu + 255 * clahe_bin_add)
        # process.show(bin_img)
        # components, label = ConnectedComponent.get_connected_component(bin_img, min_area)
        # components = process.get_near_component(components, point, max_dis)
        # components = [
        #     component for component in components
        #     if not point in component
        # ]
        # process.show(components)
        # if len(components) > 2:
        #     raise TooManyException('Peduncles are too many.')
        # elif len(components) < 2:
        #     raise TooLessException('Peduncles are too less.')

        # if test:
        #     test_data = {
        #         'clahe_img': clahe_img,
        #         'bin_img': bin_img,
        #         'otsu': otsu,
        #         'label': label
        #     }
        #     return components, test_data
        # return components
        componets = [
            component for component in self.scp_components[self.scp_slice_num]
            if process.get_distance(component.centroid, self.quad_point) > 5
            ]
        return componets

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

    def scp_combine(self, img, scps):
        mask = functools.reduce(lambda a, b: b + a, scps)
        return process.add_mask(img, mask.img_bool if hasattr(mask, 'img') else mask)

    def sitk_add(self, canvas):
        slices = [slice(dim) for dim in canvas.shape]
        slices[self.para['get_slices']['dim']] = self.real_scp_slice_num
        mask = self.scp_peduncles[0].img_bool + self.scp_peduncles[1].img_bool
        scp_slice = canvas[slices][mask] = 3.0
