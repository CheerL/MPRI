import numpy as np

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

    def in_range(self, point, strict=False):
        if strict:
            return self.img[point]
        else:
            return self.up <= point[0] < self.down and self.left <= point[1] < self.right
