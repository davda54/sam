import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from utility.cutout import Cutout
from utility.cifar_utils import fine_labels, coarse_labels, coarse_idxs, coarse_label_map
import numpy as np

class CifarHundred:
    def __init__(self, batch_size, threads, fine_granularity):
        mean, std = self._get_statistics()

        train_transform = transforms.Compose(
            [
                torchvision.transforms.RandomCrop(size=(32, 32), padding=4),
                torchvision.transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
                Cutout(),
            ]
        )

        test_transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean, std)]
        )

        train_set = torchvision.datasets.CIFAR100(
            root="./data", train=True, download=True, transform=train_transform
        )
        test_set = torchvision.datasets.CIFAR100(
            root="./data", train=False, download=True, transform=test_transform
        )

        if fine_granularity:
            self.classes = fine_labels
        else:
            train_set.classes, test_set.classes = coarse_labels, coarse_labels
            train_set.class_to_idx, test_set.class_to_idx = coarse_idxs, coarse_idxs
            train_set.targets = list(map(coarse_label_map.get, train_set.targets))
            test_set.targets = list(map(coarse_label_map.get, test_set.targets))
            self.classes = coarse_labels

        self.train = torch.utils.data.DataLoader(
            train_set, batch_size=batch_size, shuffle=True, num_workers=threads
        )
        self.test = torch.utils.data.DataLoader(
            test_set, batch_size=batch_size, shuffle=False, num_workers=threads
        )
    def _get_statistics(self):
        train_set = torchvision.datasets.CIFAR100(
            root="./cifar100",
            train=True,
            download=True,
            transform=transforms.ToTensor(),
        )

        data = torch.cat([d[0] for d in DataLoader(train_set)])
        return data.mean(dim=[0, 2, 3]), data.std(dim=[0, 2, 3])
