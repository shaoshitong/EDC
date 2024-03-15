import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.datasets import ImageFolder
from torchvision.utils import save_image
import torch.nn.functional as F
import os
import numpy as np
import warnings

warnings.filterwarnings("ignore")

# Values borrowed from https://github.com/VICO-UoE/DatasetCondensation/blob/master/utils.py
IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')
MEANS = {'cifar': [0.4914, 0.4822, 0.4465], 'imagenet': [0.485, 0.456, 0.406]}
STDS = {'cifar': [0.2023, 0.1994, 0.2010], 'imagenet': [0.229, 0.224, 0.225]}
MEANS['cifar10'] = MEANS['cifar']
STDS['cifar10'] = STDS['cifar']
MEANS['cifar100'] = MEANS['cifar']
STDS['cifar100'] = STDS['cifar']
MEANS['svhn'] = [0.4377, 0.4438, 0.4728]
STDS['svhn'] = [0.1980, 0.2010, 0.1970]
MEANS['mnist'] = [0.1307]
STDS['mnist'] = [0.3081]
MEANS['fashion'] = [0.2861]
STDS['fashion'] = [0.3530]

CLASS10 = ['n02869837','n01749939','n02488291',
            'n02107142','n13037406','n02091831',
            'n04517823','n04589890','n03062245',
            'n01773797','n01735189','n07831146',
            'n07753275','n03085013','n04485082',
            'n02105505','n01983481','n02788148',
            'n03530642','n04435653']

class ImageNetwithClass(datasets.DatasetFolder):
    def __init__(self,
                 root,
                 transform=None,
                 target_transform=None,
                 loader=datasets.folder.default_loader,
                 is_valid_file=None,
                 load_memory=False,
                 load_transform=None,
                 nclass=10,
                 phase=0,
                 slct_type='random',
                 ipc=-1,
                 seed=0):
        self.extensions = IMG_EXTENSIONS if is_valid_file is None else None
        super(ImageNetwithClass, self).__init__(root,
                                          loader,
                                          self.extensions,
                                          transform=transform,
                                          target_transform=target_transform,
                                          is_valid_file=is_valid_file)

        # Override
        if nclass < 1000:
            self.classes, self.class_to_idx = self.find_subclasses(nclass=nclass,
                                                                   phase=phase,
                                                                   seed=seed)
        else:
            self.classes, self.class_to_idx = self.find_classes(self.root)
        self.nclass = nclass
        self.samples = datasets.folder.make_dataset(self.root, self.class_to_idx, self.extensions,
                                                    is_valid_file)

        if ipc > 0:
            self.samples = self._subset(slct_type=slct_type, ipc=ipc)

        self.targets = [s[1] for s in self.samples]
        self.load_memory = load_memory
        self.load_transform = load_transform
        if self.load_memory:
            self.imgs = self._load_images(load_transform)
        else:
            self.imgs = self.samples

    def find_subclasses(self, nclass=100, phase=0, seed=0):
        """Finds the class folders in a dataset.
        """
        classes = []
        phase = max(0, phase)
        cls_from = nclass * phase
        cls_to = nclass * (phase + 1)
        if seed == 0:
            classes = CLASS10[cls_from:cls_to]
        else:
            np.random.seed(seed)
            class_indices = np.random.permutation(len(self.classes))[cls_from:cls_to]
            for i in class_indices:
                classes.append(self.classes[i])

        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        assert len(classes) == nclass

        return classes, class_to_idx

    def _subset(self, slct_type='random', ipc=10):
        n = len(self.samples)
        idx_class = [[] for _ in range(self.nclass)]
        for i in range(n):
            label = self.samples[i][1]
            idx_class[label].append(i)

        min_class = np.array([len(idx_class[c]) for c in range(self.nclass)]).min()
        print("# examples in the smallest class: ", min_class)
        assert ipc < min_class

        if slct_type == 'random':
            indices = np.arange(n)
        else:
            raise AssertionError(f'selection type does not exist!')

        samples_subset = []
        idx_class_slct = [[] for _ in range(self.nclass)]
        for i in indices:
            label = self.samples[i][1]
            if len(idx_class_slct[label]) < ipc:
                idx_class_slct[label].append(i)
                samples_subset.append(self.samples[i])

            if len(samples_subset) == ipc * self.nclass:
                break

        return samples_subset

    def _load_images(self, transform=None):
        """Load images on memory
        """
        imgs = []
        for i, (path, _) in enumerate(self.samples):
            sample = self.loader(path)
            if transform != None:
                sample = transform(sample)
            imgs.append(sample)
            if i % 100 == 0:
                print(f"Image loading.. {i}/{len(self.samples)}", end='\r')

        print(" " * 50, end='\r')
        return imgs

    def __getitem__(self, index):
        if not self.load_memory:
            path = self.samples[index][0]
            sample = self.loader(path)
        else:
            sample = self.imgs[index]

        target = self.targets[index]
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target