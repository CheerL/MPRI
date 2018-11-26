import cv2
import matplotlib.pyplot as plt
import numpy as np

class ImageProcess(object):
    @staticmethod
    def get_otsu(img, only_otsu=True):
        otsu, otsu_img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if only_otsu:
            return otsu
        return otsu, otsu_img

    @staticmethod
    def get_binary_image(img, threshold):
        _, binary_image = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
        return binary_image

    @staticmethod
    def get_grav_center(img):
        moments = cv2.moments(img)
        grav_center_x = int(moments['m10']/moments['m00'])
        grav_center_y = int(moments['m01']/moments['m00'])
        return grav_center_y, grav_center_x

    @staticmethod
    def get_clahe_image(img, limit, row, col):
        height, width = img.shape
        clahe = cv2.createCLAHE(limit * 255, (int(height/row), int(width/col)))
        return clahe.apply(img)

    @staticmethod
    def get_distance(point_1, point_2):
        return ((point_1[0] - point_2[0]) ** 2 + (point_1[1] - point_2[1]) ** 2) ** (1/2)

    @staticmethod
    def get_connected_component(img, min_area=0, sort=True):
        _, label, stats, centroids = cv2.connectedComponentsWithStats(img)
        info = [
            (level, stat, centroid)
            for level, (stat, centroid) in enumerate(zip(stats, centroids))
            if stat[4] > min_area and level != 0
        ]
        if sort:
            info.sort(key=lambda x: x[1][4], reverse=True)
        return label, info

    @staticmethod
    def get_area_distance(img, point):
        area_points = list(zip(*np.where(img)))
        distances = [ImageProcess.get_distance(area_point, point) for area_point in area_points]
        return np.array(distances).min()

    @staticmethod
    def get_revese_point(point):
        return point[1], point[0]

    @staticmethod
    def add_points(img, points, size=5, color=127, reverse=True):
        def is_point(point):
            return (isinstance(point, (tuple, list, np.ndarray)) and
                    len(point) == 2 and
                    isinstance(point[0], int) and
                    isinstance(point[1], int))

        if is_point(points):
            points = [points]
        else:
            for point in points:
                assert is_point(point)

        background = img.copy()
        for point in points:
            if reverse:
                point = ImageProcess.get_revese_point(point)
            backgorund = cv2.circle(background, point, size, color, -1)
        return backgorund

    @staticmethod
    def show(imgs, dim=0, size=6):
        if isinstance(imgs, np.ndarray):
            if len(imgs.shape) == 2:
                imgs = [imgs]
            elif len(imgs.shape) == 3:
                imgs = np.rollaxis(imgs, dim) if dim != 0 else imgs
            else:
                raise TypeError()
        assert isinstance(imgs, (list, tuple, np.ndarray))
        img_num = len(imgs)
        col = min(img_num, 2)
        row = np.ceil(img_num / col).astype(dtype=np.int32)
        _, axes = plt.subplots(row, col, figsize=(size * col, size * row))
        axes = axes.reshape(-1) if isinstance(axes, np.ndarray) else [axes]
        for img, ax in zip(imgs, axes):
            assert isinstance(img, np.ndarray)
            ax.imshow(img, 'gray')

    @staticmethod
    def get_angle(point_1, point_2):
        return point_1[0] - point_2[0], point_1[1] - point_2[1]

    @staticmethod
    def get_vert_angle(point_1, point_2, negative=False):
        angle_y, angle_x = ImageProcess.get_angle(point_1, point_2)
        signed = angle_y / abs(angle_y)

        if negative:
            return signed * angle_x, -1 * signed * angle_y
        else:
            return -1 * signed * angle_x, signed * angle_y

    @staticmethod
    def get_point_by_angle_and_distance(point, angle, distance):
        point_y, point_x = point
        angle_y, angle_x = angle
        angle_distance = (angle_x ** 2 + angle_y ** 2) ** (1 / 2)
        diff_x = distance / angle_distance * angle_x
        diff_y = distance / angle_distance * angle_y
        return (int(point_y + diff_y), int(point_x + diff_x))
