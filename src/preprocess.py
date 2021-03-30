import os
from PIL import Image
import warnings

from torch.utils.data import Dataset
import numpy as np
import bisect

from torchvision import transforms
import torch


def image_train(resize_size=256, crop_size=224):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return transforms.Compose([
        ResizeImage(resize_size),
        transforms.RandomResizedCrop(crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])


def image_test(resize_size=256, crop_size=224):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    start_center = (resize_size - crop_size - 1) / 2
    return transforms.Compose([
        ResizeImage(resize_size),
        PlaceCrop(crop_size, start_center, start_center),
        transforms.ToTensor(),
        normalize
    ])


class ResizeImage:
    def __init__(self, size):
      if isinstance(size, int):
        self.size = (int(size), int(size))
      else:
        self.size = size

    def __call__(self, img):
      th, tw = self.size
      return img.resize((th, tw))


class PlaceCrop:
    """Crops the given PIL.Image at the particular index.
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (w, h), a square crop (size, size) is
            made.
    """

    def __init__(self, size, start_x, start_y):
        if isinstance(size, int):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.start_x = start_x
        self.start_y = start_y

    def __call__(self, img):
        """
        Args:
            img (PIL.Image): Image to be cropped.
        Returns:
            PIL.Image: Cropped image.
        """
        th, tw = self.size
        return img.crop((self.start_x, self.start_y, self.start_x + tw, self.start_y + th))


class ConcatDataset(Dataset):
    """
    Dataset to concatenate multiple datasets.
    Purpose: useful to assemble different existing datasets, possibly
    large-scale datasets as the concatenation operation is done in an
    on-the-fly manner.

    Arguments:
        datasets (sequence): List of datasets to be concatenated
    """

    @staticmethod
    def cumsum(sequence):
        r, s = [], 0
        for e in sequence:
            l = len(e)
            r.append(l + s)
            s += l
        return r

    def __init__(self, datasets):
        super(ConcatDataset, self).__init__()
        assert len(datasets) > 0, 'datasets should not be an empty iterable'
        self.datasets = list(datasets)
        self.cumulative_sizes = self.cumsum(self.datasets)

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][sample_idx]

    @property
    def cummulative_sizes(self):
        warnings.warn('cummulative_sizes attribute is renamed to '
                      'cumulative_sizes', DeprecationWarning, stacklevel=2)
        return self.cumulative_sizes


"""
Code adapted from CDAN github repository.
https://github.com/thuml/CDAN/tree/master/pytorch
"""


class ImageList(Dataset):
    def __init__(self, image_root, image_list_root, dataset, domain_label, dataset_name, split='train', transform=None, 
                 sample_masks=None, pseudo_labels=None):
        self.image_root = image_root
        self.dataset = dataset  # name of the domain
        self.dataset_name = dataset_name  # name of whole dataset
        self.transform = transform
        self.loader = self._rgb_loader
        self.sample_masks = sample_masks
        self.pseudo_labels = pseudo_labels
        if dataset_name == 'domain-net':
            imgs = self._make_dataset(os.path.join(image_list_root, dataset + '_' + split + '.txt'), domain_label)
        else:
            imgs = self._make_dataset(os.path.join(image_list_root, dataset + '.txt'), domain_label)
        self.imgs = imgs
        if sample_masks is not None:
            temp_list = self.imgs
            self.imgs = [temp_list[i] for i in self.sample_masks]
            if pseudo_labels is not None:
                self.labels = self.pseudo_labels[self.sample_masks]
                assert len(self.labels) == len(self.imgs), 'Lengths do no match!'

    def _rgb_loader(self, path):
        with open(path, 'rb') as f:
            with Image.open(f) as img:
                return img.convert('RGB')

    def _make_dataset(self, image_list_path, domain):
        image_list = open(image_list_path).readlines()
        images = [(val.split()[0], int(val.split()[1]), int(domain)) for val in image_list]
        return images

    def __getitem__(self, index):
        output = {}
        path, target, domain = self.imgs[index]
        if self.dataset_name == 'domain-net':
            img = self.loader(os.path.join(self.image_root, path))
        elif self.dataset_name in ['office-home', 'pacs']:
            img = self.loader(os.path.join(self.image_root, self.dataset, path))
        elif self.dataset_name == 'office31':
            img = self.loader(os.path.join(self.image_root, self.dataset, 'images', path))
        if self.transform is not None:
            img = self.transform(img)
        
        output['img'] = img
        if self.pseudo_labels is not None:
            output['target'] = torch.squeeze(torch.LongTensor([np.int64(self.labels[index]).item()]))
        else:
            output['target'] = torch.squeeze(torch.LongTensor([np.int64(target).item()]))
        output['domain'] = domain
        output['idx'] = index

        return output

    def __len__(self):
        return len(self.imgs)
