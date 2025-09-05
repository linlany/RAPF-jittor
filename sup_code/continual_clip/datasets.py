

import os
import pdb
# import torch.nn as nn
import jittor as jt
import jittor.nn as nn
import numpy as np
from typing import Tuple, Union
# from continuum import ClassIncremental, InstanceIncremental
# from continuum.datasets import (
#     CIFAR100, ImageNet100, TinyImageNet200, ImageFolderDataset, Core50
# )
from .class_incremental import ClassIncremental
from .base import ImageFolderDataset, _ContinuumDataset, TaskType
from .utils import get_dataset_class_names, get_workdir
from .cifar100 import CIFAR100

# from torchvision import transforms
import jittor.transform as transforms

from jittor.dataset import Dataset
from PIL import Image
from jittor_utils import LOG



class MyImageFolder(Dataset):
    """
    A image classify dataset, load image and label from directory::
    
        * root/label1/img1.png
        * root/label1/img2.png
        * ...
        * root/label2/img1.png
        * root/label2/img2.png
        * ...

    Args::

        [in] root(string): Root directory path.
        [in] data(object): Data object containing _x (paths), _y (labels), and _t (task ids).

    Attributes::

        * classes(list): List of the class names.
        * class_to_idx(dict): map from class_name to class_index.
        * imgs(list): List of (image_path, class_index) tuples
        * taskid(list): List of task ids

    Example::

        train_dir = './data/celebA_train'
        train_loader = ImageFolder(train_dir).set_attrs(batch_size=batch_size, shuffle=True)
        for batch_idx, (x_, target) in enumerate(train_loader):
            ...

    """
    def __init__(self, root=None, data=None, transform=None):
        super().__init__()
        self.transform = transform
        self.imgs = []
        self.taskid = []

        if data is not None:
            self.imgs = list(zip(map(str, data._x), map(int, data._y)))
            # pdb.set_trace()
            self.taskid = data._t
            self.set_attrs(total_len=len(self.imgs))
        elif root is not None:
            self.root = root
            self.classes = sorted([d.name for d in os.scandir(root) if d.is_dir()])
            self.class_to_idx = {v:k for k,v in enumerate(self.classes)}
            image_exts = set(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'))
            
            for i, class_name in enumerate(self.classes):
                class_dir = os.path.join(root, class_name)
                for dname, _, fnames in sorted(os.walk(class_dir, followlinks=True)):
                    for fname in sorted(fnames):
                        if os.path.splitext(fname)[-1].lower() in image_exts:
                            path = os.path.join(class_dir, fname)
                            self.imgs.append((path, i))
            LOG.i(f"Found {len(self.classes)} classes and {len(self.imgs)} images.")
            self.set_attrs(total_len=len(self.imgs))
        
    def __getitem__(self, k):
        with open(self.imgs[k][0], 'rb') as f:
            img = Image.open(f).convert('RGB')
            if self.transform:
                img = self.transform(img)
            return img, self.imgs[k][1], self.taskid[k]


class ImageNet100(_ContinuumDataset):
    """Subset of ImageNet1000 made of only 100 classes.

    You must download the ImageNet1000 dataset then provide the images subset.
    If in doubt, use the option at initialization `download=True` and it will
    auto-download for you the subset ids used in:
        * Small Task Incremental Learning
          Douillard et al. 2020
    """

    train_subset_url = "https://github.com/Continvvm/continuum/releases/download/v0.1/train_100.txt"
    test_subset_url = "https://github.com/Continvvm/continuum/releases/download/v0.1/val_100.txt"

    def __init__(
            self, *args, data_subset: Union[Tuple[np.array, np.array], str, None] = None, **kwargs
    ):
        self.data_subset = data_subset
        super().__init__(*args, **kwargs)

    @property
    def data_type(self) -> TaskType:
        return TaskType.IMAGE_PATH

    @property
    def transformations(self):
        """Default transformations if nothing is provided to the scenario."""
        # return [
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            transforms.ImageNormalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        # ]
        ])

    # def _download(self):
    #     if not os.path.exists(self.data_path):
    #         raise IOError(
    #             "You must download yourself the ImageNet dataset."
    #             " Please go to http://www.image-net.org/challenges/LSVRC/2012/downloads and"
    #             " download 'Training images (Task 1 & 2)' and 'Validation images (all tasks)'."
    #         )
    #     print("ImageNet already downloaded.")

    #     filename = "val_100.txt"
    #     self.subset_url = self.test_subset_url
    #     if self.train:
    #         filename = "train_100.txt"
    #         self.subset_url = self.train_subset_url

    #     if self.data_subset is None:
    #         self.data_subset = os.path.join(self.data_path, filename)
    #         if not os.path.exists(self.data_subset):
    #             print("Downloading subset indexes...", end=" ")
    #             download(self.subset_url, self.data_path)
    #             print("Done!")

    def get_data(self) -> Tuple[np.ndarray, np.ndarray, Union[np.ndarray, None]]:
        data = self._parse_subset(self.data_subset, train=self.train)  # type: ignore
        return (*data, None)

    def _parse_subset(
            self,
            subset: Union[Tuple[np.array, np.array], str, None],
            train: bool = True
    ) -> Tuple[np.array, np.array]:
        if isinstance(subset, str):
            x, y = [], []

            with open(subset, "r") as f:
                for line in f:
                    split_line = line.split(" ")
                    path = split_line[0].strip()
                    x.append(os.path.join(self.data_path, path))
                    y.append(int(split_line[1].strip()))
            x = np.array(x)
            y = np.array(y)
            return x, y
        return subset  # type: ignore

class ImageNet1000(ImageFolderDataset):
    """Continuum dataset for datasets with tree-like structure.
    :param train_folder: The folder of the train data.
    :param test_folder: The folder of the test data.
    :param download: Dummy parameter.
    """

    def __init__(
            self,
            data_path: str,
            train: bool = True,
            download: bool = False,
    ):
        super().__init__(data_path=data_path, train=train, download=download)

    def get_data(self):
        if self.train:
            self.data_path = os.path.join(self.data_path, "train")
        else:
            self.data_path = os.path.join(self.data_path, "val")
        return super().get_data()


class ImageNet_R(ImageFolderDataset):
    """Continuum dataset for datasets with tree-like structure.
    :param train_folder: The folder of the train data.
    :param test_folder: The folder of the test data.
    :param download: Dummy parameter.
    """

    def __init__(
            self,
            data_path: str,
            train: bool = True,
            download: bool = False,
    ):
        super().__init__(data_path=data_path, train=train, download=download)
    @property
    def transformations(self):
        """Default transformations if nothing is provided to the scenario."""
        # return [
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            transforms.ImageNormalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        # ]
        ])

    def get_data(self):
        if self.train:
            self.data_path = os.path.join(self.data_path, "train")
        else:
            self.data_path = os.path.join(self.data_path, "test")
        return super().get_data()


def get_dataset(cfg, is_train, transforms=None):
    if cfg.dataset == "cifar100":
        # data_path = os.path.join(cfg.dataset_root, cfg.dataset)
        data_path = cfg.dataset_root
        dataset = CIFAR100(
            data_path=data_path, 
            download=True, 
            train=is_train, 
            # transforms=transforms
        )
        # classes_names = dataset.dataset.classes
        classes_names = dataset.classes
    elif cfg.dataset == "imagenet_R":
        data_path = cfg.dataset_root
        dataset = ImageNet_R(
            data_path, 
            train=is_train
        )
        classes_names = get_dataset_class_names(cfg.workdir, cfg.dataset)
        
    elif cfg.dataset == "imagenet100":
        data_path = cfg.dataset_root
        dataset = ImageNet100(
            data_path, 
            train=is_train,
            data_subset=os.path.join(get_workdir(os.getcwd()), "class_orders/train_100.txt" if is_train else "class_orders/val_100.txt")
        )
        classes_names = get_dataset_class_names(cfg.workdir, cfg.dataset)

    elif cfg.dataset == "imagenet1000":
        data_path = os.path.join(cfg.dataset_root, cfg.dataset)
        dataset = ImageNet1000(
            data_path, 
            train=is_train
        )
        classes_names = get_dataset_class_names(cfg.workdir, cfg.dataset)
    
    else:
        ValueError(f"'{cfg.dataset}' is a invalid dataset.")

    return dataset, classes_names


def build_cl_scenarios(cfg, is_train, transforms) -> nn.Module:

    dataset, classes_names = get_dataset(cfg, is_train)
    # pdb.set_trace()
    if cfg.scenario == "class":
        scenario = ClassIncremental(
            dataset,
            initial_increment=cfg.initial_increment,
            increment=cfg.increment,
            transformations=transforms.transforms, # Convert Compose into list
            class_order=cfg.class_order,
        )

    else:
        ValueError(f"You have entered `{cfg.scenario}` which is not a defined scenario, " 
                    "please choose from {{'class', 'domain', 'task-agnostic'}}.")

    return scenario, classes_names