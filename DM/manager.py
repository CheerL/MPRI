import os
import random
from collections import deque

from config import DATA_PATH

class FileManager(object):
    def __init__(self, path, file_type):
        self.path = path
        self.file_type = file_type

    def load(self):
        raise NotImplementedError()

class DataManager(object):
    def __init__(self, data_path=DATA_PATH):
        self.data_path = data_path
        self.file_list = list()

    def create_file_list(self, limit=0):
        ''' find all directories containing images and put there name in the file_list'''
        if self.file_list:
            self.file_list.clear()

        stack = deque([self.data_path])

        while stack:
            now_path = stack.pop()
            if os.path.isdir(now_path):
                stack.extend([os.path.join(now_path, sub)
                              for sub in os.listdir(now_path)])
            else:
                if now_path.endswith('nii.gz') or now_path.endswith('nii'):
                    now_dir = os.path.dirname(now_path)
                    if now_dir not in self.file_list:
                        self.file_list.append(now_dir)

        self.check_file_list()
        if limit > 0:
            self.file_list = random.sample(self.file_list, k=limit)

    def check_file_list(self):
        for img_dir in self.file_list:
            img_str = ''.join(os.listdir(img_dir))
            for mod in self.mods:
                if mod not in img_str:
                    self.file_list.remove(img_dir)
                    break

    def load(self, file_list=None):
        ''' load images from image.nii'''
        raise NotImplementedError()

    def split_data(self, val_rate=None):
        if not self.file_list:
            self.create_file_list()

        if val_rate is None:
            val_rate = self.val_rate

        total_num = len(self.file_list)
        val_num = int(total_num * val_rate)
        val_num_list = random.sample(range(total_num), val_num)
        self.val_list = [path for i, path in enumerate(
            self.file_list) if i in val_num_list]
        self.train_list = [path for i, path in enumerate(
            self.file_list) if i not in val_num_list]
