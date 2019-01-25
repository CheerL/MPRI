import os
import random
from functools import wraps

import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk


def nii_check_loaded(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        assert self.loaded, 'nii file not loaded'
        return func(self, *args, **kwargs)
    return wrapper


class FileManager(object):
    def __init__(self, path, file_type):
        self.path = path
        self.file_type = file_type

    def load(self):
        raise NotImplementedError()


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

    def normalize(self):
        self.img = self.img / self.img.max() * 255
        self.img = self.img.astype(np.uint8)

    @nii_check_loaded
    def get_slice(self, start, end=None, dim=0):
        assert isinstance(dim, int) and dim in [0, 1, 2]
        assert isinstance(start, int)
        assert (
            start < self.size[dim]
            if end is None else
            (isinstance(end, int) and start <= end <= self.size[dim])
        )

        img_slice = self.img if dim is 0 else np.rollaxis(self.img, dim)
        return img_slice[start] if end is None else img_slice[start:end]

    @nii_check_loaded
    def show(self, pos, dim=0):
        plt.imshow(self.get_slice(pos, dim=dim), 'gray')


class RotatedNiiFileManager(NiiFileManager):
    def load(self):
        self.sitk_img = sitk.ReadImage(self.path)
        self.img = np.rot90(
            sitk.GetArrayFromImage(
                self.sitk_img
            ).astype(np.float32),
            k=2, axes=(0, 1)
        )
        self.size = self.img.shape
        self.loaded = True

class LabelNiiFileManager(RotatedNiiFileManager):
    def get_label(self, label, label_types):
        if isinstance(label_types, int):
            return label==label_types
        elif isinstance(label_types, (list, tuple)):
            return np.logical_or(*[
                label==label_type for label_type in label_types
            ])

if __name__ == '__main__':
    pass
