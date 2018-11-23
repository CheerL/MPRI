import os
import random
from functools import wraps

import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk

from DM.manager import DataManager, FileManager


def nii_check_loaded(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        assert self.loaded, 'nii file not loaded'
        return func(self, *args, **kwargs)
    return wrapper

class NiiFileManager(FileManager):
    def __init__(self, path):
        super().__init__(path, 'nii')
        self.loaded = False
        self.size = None
        self.sitk_img = None
        self.img = None

    def load(self):
        self.sitk_img = sitk.ReadImage(self.path)
        self.size = self.sitk_img.GetSize()
        self.img = sitk.GetArrayFromImage(self.sitk_img).astype(np.float32)
        self.loaded = True

    @nii_check_loaded
    def get_slice(self, start, end=None, dim=0):
        assert isinstance(dim, int) and dim in [0, 1, 2]
        assert isinstance(start, int)
        assert (
            start < self.size[dim]
            if end is None else
            (isinstance(end, int) and start <= end <= self.size[dim])
        )

        if dim is 0:
            img_slice = self.img[start, :, :] if end is None else self.img[start:end, :, :]
        elif dim is 1:
            img_slice = self.img[:, start, :] if end is None else self.img[:, start:end, :]
        else:
            img_slice = self.img[:, :, start] if end is None else self.img[:, :, start:end]
        return img_slice

    @nii_check_loaded
    def show(self, pos, dim=0):
        plt.imshow(self.get_slice(pos, dim=dim), 'gray')

class DataManagerNii(DataManager):
    pass

if __name__ == '__main__':
    pass
