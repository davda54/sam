import argparse
import os
import pickle
from pathlib import Path

import pandas as pd
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from model.wide_res_net import WideResNet
from utility.cifar_utils import cifar100_stats

torch.multiprocessing.set_sharing_strategy("file_system")


def get_project_path() -> Path:
    return Path(__file__).parent.parent


def get_granularity(name: str) -> str:
    if "coarse" in name:
        return "coarse"
    elif "fine" in name:
        return "fine"
    else:
        raise ValueError("granularity not found")


def get_parameter(name: str, param: str) -> int:
    extension = "." + name.split(".")[-1]
    if param not in ["crop", "kernel", "width", "depth"]:
        raise ValueError("invalid parameter input")
    for element in name.split("_"):
        if param in element:
            return int(element.replace(param, "").replace(extension, ""))


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
    dataset_type = dataloader.dataset.cifar100.meta["type"]
    results = {}
    with torch.no_grad():
        for inputs, targets, idxs in tqdm(
            dataloader, desc=f"Evaluating {dataset_type} data"
        ):
            # TODO: Determine if using cuda device speeds things up here
            # inputs, targets, idxs = (b.to(device) for b in batch)
            # print(f"Batch idx {batch_idx}, dataset index {idxs}")
            outputs = model(inputs)
            predictions = torch.argmax(outputs, 1)
            correct = torch.argmax(outputs, 1) == targets
            zipped = zip(idxs, zip(*(outputs, predictions, targets, correct,)))
            for idx, data in zipped:
                results[idx] = data
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
        test_dataset, batch_size=128, shuffle=False, num_workers=2,
    )

    return test_dataloader


def get_validation_dataloader():
    validation_dataset = torch.load(
        dataset_path / "validation" / "validation_dataset.pt"
    )
    validation_dataset.cifar100.meta["type"] = "validation"
    validation_dataloader = torch.utils.data.DataLoader(
        validation_dataset, batch_size=128, shuffle=False, num_workers=2,
    )
    return validation_dataloader


def main(_args):
    """
    from the training set, get 100 images for each superclass
    export as validation set
    """
    device = torch.device(f"cuda:{_args.gpu}" if torch.cuda.is_available() else "cpu")

    test_dataloader = get_test_dataloader()
    validation_dataloader = get_validation_dataloader()

    model_paths = find_model_files()

    # TODO: Remove later CROPPING DOWN PATHS DURING DEVELOPMENT
    model_paths = model_paths[:2]
    model_results = {}

    # TODO: dump results to file after each iteration in this loop
    # TODO: speed this by increasing batch size or use gpus/multiprocessing
    for model_path in tqdm(model_paths, desc="Model evaluations"):
        model_filename = str(model_path.split("/")[-1])

        crop_size = int(get_parameter(model_filename, "crop"))
        kernel_size = int(get_parameter(model_filename, "kernel"))
        width_factor = int(get_parameter(model_filename, "width"))
        depth = int(get_parameter(model_filename, "depth"))
        granularity = get_granularity(model_filename)

        if granularity == "coarse":
            n_labels = 20
        elif granularity == "fine":
            n_labels = 100
        else:
            raise ValueError("model filename does not contain granularity")
        model = WideResNet(
            kernel_size=kernel_size,
            width_factor=width_factor,
            depth=depth,
            dropout=0.0,
            in_channels=3,
            labels=n_labels,
        )
        model_state_dict = torch.load(model_path, map_location=f"cuda:{_args.gpu}")[
            "model_state_dict"
        ]
        model.load_state_dict(model_state_dict)
        model.eval()
        validation_results = evaluate(validation_dataloader, model, device)
        test_results = evaluate(test_dataloader, model, device)
        model_results[model_filename] = {}
        model_results[model_filename]["validation"] = validation_results
        model_results[model_filename]["test"] = test_results

    return model_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gpu", default=6, type=int, help="Index of GPU to use",
    )
    args = parser.parse_args()
    print("Getting model results")
    model_results = main(args)
    print("Pickling results to file")
    pickle.dump(model_results, open(str(project_path / "model_results.pkl"), "wb"))
    # save to CSV using pandas
