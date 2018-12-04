import cv2
import numpy as np
from process import base_process


class ConnectedComponent(object):
    def __init__(self, img, level, stat, centroid):
        self.img = img
        self.shape = img.shape
        self.level = level
        self.left = stat[0]
        self.right = stat[0] + stat[2]
        self.up = stat[1]
        self.down = stat[1] + stat[3]
        self.area = stat[4]
        self.centroid = centroid

    def __lt__(self, other):
        return self.area < other.area

    def __le__(self, other):
        return self.area <= other.area

    @staticmethod
    def get_connected_component(img, min_area=0, sort=True):
        _, label, stats, centroids = cv2.connectedComponentsWithStats(img)
        components = [
            ConnectedComponent(
                label == level, level, stat,
                base_process.get_revese_point(centroid.astype(np.int))
            ) for level, (stat, centroid) in enumerate(zip(stats, centroids))
            if stat[4] > min_area and level != 0
        ]
        if sort:
            components = sorted(components, reverse=True)
        return components, label

    @property
    def img_uint8(self):
        if self.img.dtype == np.uint8:
            return self.img
        return (self.img * 255 / self.img.max()).astype(np.uint8)

    @property
    def img_bool(self):
        if self.img.dtype == bool:
            return self.img
        return self.img.astype(bool)

    @property
    def canvas(self, dtype=np.uint8):
        return np.zeros(self.shape).astype(dtype)

    def get_bound_point(self, type_='l'):
        y_pos, x_pos = np.where(self.img)
        if type_ == 'l':
            left_y = int(y_pos[np.where(x_pos == self.left)[0]].mean())
            bound_point = (left_y, self.left)
        elif type_ == 'r':
            right_y = int(y_pos[np.where(x_pos == self.right-1)[0]].mean())
            bound_point = (right_y, self.right-1)
        elif type_ == 'u':
            up_x = int(x_pos[np.where(y_pos == self.up)[0]].mean())
            bound_point = (self.up, up_x)
        elif type_ == 'd':
            down_x = int(x_pos[np.where(y_pos == self.down-1)[0]].mean())
            bound_point = (self.down-1, down_x)
        elif 1 < len(type_) <= 4:
            bound_point = [self.get_bound_point(each_type_) for each_type_ in type_]
        else:
            raise TypeError()
        return bound_point

    def __contains__(self, point):
        if base_process.is_point(point):
            return self.up <= point[0] < self.down and self.left <= point[1] < self.right
        return False

    def in_range(self, left=None, right=None, up=None, down=None):
        if left is not None and left > self.left:
            return False
        if right is not None and right < self.right:
            return False
        if up is not None and up > self.up:
            return False
        if down is not None and down < self.down:
            return False
        return True
