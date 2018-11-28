class ConnectedComponent(object):
    def __init__(self, img, level, stat, centroid):
        self.img = img
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
