import argparse
import os
import pickle
from pathlib import Path

import pandas as pd
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset

from model.wide_res_net import WideResNet
from utility.cifar_utils import cifar100_stats


def get_project_path() -> Path:
    return Path(__file__).parent.parent


class CIFAR100Indexed(Dataset):
    def __init__(self, root, download, train, transform):
        self.cifar100 = torchvision.datasets.CIFAR100(
            root=root, download=download, train=train, transform=transform
        )

    def __getitem__(self, index):
        data, target = self.cifar100[index]
        return data, target, index

    def __len__(self):
        return len(self.cifar100)


project_path = get_project_path()
dataset_path = project_path / "datasets"
dataset_path.mkdir(parents=True, exist_ok=True)


def find_model_files(model_path=(project_path / "models")):
    model_files = []
    for root, directories, files in os.walk(model_path):
        for file in files:
            if file.startswith("model_") and file.endswith(".pt"):
                model_files.append(os.path.join(root, file))
    return model_files


def evaluate(dataloader, model, device):
    results = {}
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            inputs, targets, idx = (b.to(device) for b in batch)
            outputs = model(inputs)
            predictions = torch.argmax(outputs, 1)
            correct = torch.argmax(outputs, 1) == targets
            results[batch_idx] = list(
                zip(idx, targets, outputs, predictions, correct)
            )  # TODO: NamedTuple? Dict?
    return results


def get_test_dataloader():
    mean, std = cifar100_stats(root=str(dataset_path))
    test_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean, std)]
    )
    test_dataset = CIFAR100Indexed(
        root=str(dataset_path), train=False, download=False, transform=test_transform,
    )

    test_dataset.cifar100.meta["type"] = "test"

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1, shuffle=False, num_workers=2,
    )

    return test_dataloader


def get_validation_dataloader():
    validation_dataset = torch.load(
        dataset_path / "validation" / "validation_dataset.pt"
    )
    validation_dataset.meta["type"] = "validation"
    validation_dataloader = torch.utils.data.DataLoader(
        validation_dataset, batch_size=1, shuffle=False, num_workers=2,
    )
    return validation_dataloader


def main(_args):
    """
    from the training set, get 100 images for each superclass
    export as validation set
    """
    device = torch.device(f"cuda:{_args.gpu}" if torch.cuda.is_available() else "cpu")

    test_dataloader = get_test_dataloader()
    # validation_dataloader = get_validation_dataloader()

    model_paths = find_model_files()

    model_results = {}

    for model_path in model_paths:
        model_filename = str(model_path.split("/")[-1])

        # TODO: Parse out the crop/kernel/width/depth from the model's filepath
        # TODO: Init the model object using the parsed values
        # TODO: Parse out crop or fine from model fp and assign n_labels
        # for element in model_fp.split("_"):
        #     if 'crop' in element:
        #         crop_size = int(element.replace('crop', ''))
        # ts, vs = copy(test_set), copy(validation_set)

        if "coarse" in model_filename:
            n_labels = 20
        elif "fine" in model_filename:
            n_labels = 100
        else:
            raise ValueError("model filename does not contain granularity")
        model = WideResNet(
            kernel_size=_args.kernel_size,
            width_factor=_args.width_factor,
            depth=_args.depth,
            dropout=0.0,
            in_channels=3,
            labels=n_labels,
        )
        model_state_dict = torch.load(model_path, map_location=f"cuda:{_args.gpu}")[
            "model_state_dict"
        ]
        model.load_state_dict(model_state_dict)
        model.eval()
        test_results = evaluate(model, test_dataloader, device)
        # validation_results = evaluate(validation_dataloader, model, device)
        model_results[
            model_filename
        ] = {}  # TODO: Get the model name and use it to store the results
        model_results[model_filename]["test"] = test_results
        # model_results[model_filename]['validation'] = validation_results

    return model_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gpu", default=6, type=int, help="Index of GPU to use",
    )
    args = parser.parse_args()
    model_results = main(args)
    pickle.dump(model_results, open(str(project_path / "model_results.pkl"), "wb"))
    # save to CSV using pandas
