import numpy as np
import pickle

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from example.utility.cutout import Cutout
from example.utility.cifar_utils import fine_labels, coarse_labels, coarse_idxs, coarse_label_map


class CifarHundred:
    def __init__(self, fine_labels, crop_size, batch_size, threads):
        mean, std = self._get_statistics()

        train_transform = transforms.Compose(
            [
                torchvision.transforms.RandomCrop(size=(crop_size, crop_size), padding=4),
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
            root="./datasets/cifar100", train=True, download=True, transform=train_transform
        )
        test_set = torchvision.datasets.CIFAR100(
            root="./datasets/cifar100", train=False, download=True, transform=test_transform
        )

        if fine_labels:
            self.classes = fine_labels
        else:
            self.classes = coarse_classes
            train_set.classes, test_set.classes = coarse_classes, coarse_classes
            train_set.class_to_idx, test_set.class_to_idx = coarse_idxs, coarse_idxs
            train_set.targets = list(map(coarse_label_map.get, train_set.targets))
            test_set.targets = list(map(coarse_label_map.get, test_set.targets))

        self.train = torch.utils.data.DataLoader(
            train_set, batch_size=batch_size, shuffle=True, num_workers=threads
        )
        self.test = torch.utils.data.DataLoader(
            test_set, batch_size=batch_size, shuffle=False, num_workers=threads
        )
    def _get_statistics(self):
        train_set = torchvision.datasets.CIFAR100(
            root="./datasets/cifar100",
            train=True,
            download=True,
            transform=transforms.ToTensor(),
        )

        data = torch.cat([d[0] for d in DataLoader(train_set)])
        return data.mean(dim=[0, 2, 3]), data.std(dim=[0, 2, 3])



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fine_labels",
        default=True,
        type=bool,
        help="True to use CIFAR100 fine granularity, False for coarse class granularity.",
    )
    parser.add_argument(
        "--crop_size",
        default=32,
        type=int,
        help="Crop size used in data transformations.",
    )
    parser.add_argument(
        "--batch_size",
        default=128,
        type=int,
        help="Batch size used in the training and validation loop.",
    )
    parser.add_argument(
        "--threads", default=2, type=int, help="Number of CPU threads for dataloaders."
    )
    args = parser.parse_args()

    print(args)

    initialize(args, seed=42)
    device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")

    dataset = CifarHundred(args.fine_classes, args.crop_size, args.batch_size, args.threads)
    dataset_filename = str(
        Path.cwd()
        / "datasets"
        / "cifar100"
        / f"cifar100data_fine{args.fine_classes}_crop{args.crop_size}_batch{args.batch_size}_threads{args.threads}.pkl"
    )
    dataset_file = open(dataset_filename, 'wb')

    pickle.dump(dataset, dataset_file)
    dataset_file.close()
