import cv2
import matplotlib.pyplot as plt
import numpy as np

from process.base_image import ImageProcess

class TooBigException(Exception):
    pass

class TooSmallExceotion(Exception):
    pass

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
        bin_img = self.get_binary_image(region_img, otsu*(1+rate))
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
                         clahe_bin_add=0.2, min_area=10, max_distance=10,
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

    def get_scp_width(self, component, test=False):
        contours = cv2.findContours(
            component.img_uint8,
            cv2.RETR_TREE,
            cv2.CHAIN_APPROX_NONE
            )
        min_rect = cv2.minAreaRect(contours[1][0])
        # box = np.int32(np.around(cv2.boxPoints(min_rect)))
        box = cv2.boxPoints(min_rect)
        width = min(
            self.get_distance(box[0], box[1]),
            self.get_distance(box[0], box[2]),
            self.get_distance(box[0], box[3])
        )
        self.show(component.img)
        print(width)
        if width > 10:
            raise TooBigException('SCP is to big')
        elif width < 2:
            raise TooSmallExceotion('SCP is to small')

        if test:
            test_data = {
                'box': np.around(box).astype(np.int),
            }
            return width, test_data
        return width
