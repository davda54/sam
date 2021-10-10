import argparse
import csv
import os

# import pickle
from pathlib import Path

import pandas as pd
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from model.wide_res_net import WideResNet
from utility.cifar_utils import (
    cifar100_stats,
    coarse_classes,
    coarse_idxs,
    fine_to_coarse_idxs,
)

torch.multiprocessing.set_sharing_strategy("file_system")
from collections import namedtuple
from itertools import compress

from utility.cifar_utils import coarse_class_to_idx

project_path = Path(__file__).parent.parent
dataset_path = project_path / "datasets"
dataset_path.mkdir(parents=True, exist_ok=True)
evaluations_path = project_path / "evaluations"
evaluations_path.mkdir(parents=True, exist_ok=True)

Result = namedtuple("Result", ["idx", "output", "prediction", "target", "correct"])
profile_fields = [
    "granularity",
    "superclass",
    "crop_size",
    "kernel_size",
    "width_factor",
    "depth",
    "accuracy",
    "flops",
]
Profile = namedtuple("Profile", profile_fields,)


def get_granularity(name: str) -> str:
    if "coarse" in name:
        return "coarse"
    elif "fine" in name:
        return "fine"
    else:
        raise ValueError("granularity not found")


def get_superclass(name: str) -> int:
    pass


def get_parameter(name: str, param: str) -> int:
    extension = "." + name.split(".")[-1]
    if param not in ["class", "crop", "kernel", "width", "depth"]:
        raise ValueError("invalid parameter input")
    for element in name.split("_"):
        if param in element:
            return int(element.replace(param, "").replace(extension, ""))


def get_parameters(model_filename):
    granularity = get_granularity(model_filename)
    class_id = int(get_parameter(model_filename, "class"))
    crop_size = int(get_parameter(model_filename, "crop"))
    kernel_size = int(get_parameter(model_filename, "kernel"))
    width_factor = int(get_parameter(model_filename, "width"))
    depth = int(get_parameter(model_filename, "depth"))
    return granularity, class_id, crop_size, kernel_size, width_factor, depth


def parse_model_path(model_path):
    model_name = str(model_path.split("/")[-1])
    model_name = model_name.replace(".pt", "").replace("model_", "")
    model_name = superclass_to_idx(model_name)
    return model_name


def superclass_to_idx(filename: str):
    """
    input a model filename
    output is model filename with the superclass label name replaced with index
    coarse granularity models have their generic term removed
    """
    if "all_" in filename:  # if coarse just removes the superclass placeholder
        return filename.replace("_all_", "_class-1_")
    keys = coarse_class_to_idx.keys()
    superclass = next(compress(keys, [k in filename for k in keys]))
    superclass_idx = coarse_class_to_idx[superclass]
    return filename.replace(superclass, "class" + str(superclass_idx))


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


def find_model_files(model_path=(project_path / "models")):
    model_files = []
    for root, directories, files in os.walk(model_path):
        for file in files:
            if file.startswith("model_") and file.endswith(".pt"):
                model_files.append(os.path.join(root, file))
    return model_files


def evaluate(dataloader, model, dataset_type):
    results = []
    # total_loss = 0.0
    total_correct = 0.0
    count = 0.0
    with torch.no_grad():
        for inputs, targets, idxs in tqdm(
            dataloader, desc=f"Evaluating {dataset_type} data"
        ):
            # TODO: Determine if using cuda device speeds things up here
            # inputs, targets, idxs = (b.to(device) for b in batch)
            # print(f"Batch idx {batch_idx}, dataset index {idxs}")
            count += len(inputs)
            outputs = model(inputs)
            # total_loss += smooth_crossentropy(outputs, targets)
            predictions = torch.argmax(outputs, 1)
            correct = torch.argmax(outputs, 1) == targets
            total_correct += correct.sum().item()
            zipped = zip(idxs, zip(*(outputs, predictions, targets, correct)),)
            for idx, data in zipped:
                result_ = [idx.tolist()] + [d.tolist() for d in data]
                results.append(Result(*result_))
    accuracy = total_correct / count
    return results, accuracy


def get_test_dataloader(coarse=False):
    mean, std = cifar100_stats(root=str(dataset_path))
    test_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean, std)]
    )
    test_dataset = CIFAR100Indexed(
        root=str(dataset_path), train=False, download=False, transform=test_transform,
    )

    test_dataset.cifar100.meta["type"] = "test"

    if coarse:
        test_dataset.cifar100.classes = coarse_classes
        test_dataset.cifar100.class_to_idx = coarse_idxs
        test_dataset.cifar100.targets = list(
            map(fine_to_coarse_idxs.get, test_dataset.cifar100.targets)
        )

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1024, shuffle=False, num_workers=10,
    )

    return test_dataloader


def get_validation_dataloader(coarse=False):
    validation_dataset = torch.load(
        dataset_path / "validation" / "validation_dataset.pt"
    )
    validation_dataset.cifar100.meta["type"] = "validation"

    if coarse:
        validation_dataset.cifar100.classes = coarse_classes
        validation_dataset.cifar100.class_to_idx = coarse_idxs
        validation_dataset.cifar100.targets = list(
            map(fine_to_coarse_idxs.get, validation_dataset.cifar100.targets)
        )

    validation_dataloader = torch.utils.data.DataLoader(
        validation_dataset, batch_size=1024, shuffle=False, num_workers=10,
    )
    return validation_dataloader


def main(_args):
    """
    from the training set, get 100 images for each superclass
    export as validation set
    """
    device = torch.device(f"cuda:{_args.gpu}" if torch.cuda.is_available() else "cpu")

    test_fine_dataloader = get_test_dataloader(coarse=False)
    test_coarse_dataloader = get_test_dataloader(coarse=True)
    validation_coarse_dataloader = get_validation_dataloader(coarse=False)
    validation_fine_dataloader = get_validation_dataloader(coarse=False)

    model_paths = find_model_files()

    model_paths = model_paths[: _args.limit]
    # model_results = {}

    profiles_path = evaluations_path / "model_profiles.csv"
    with open(profiles_path, "w", encoding="UTF8") as f:
        writer = csv.writer(f)
        writer.writerow(profile_fields)

    # TODO: can I speed this up using gpus/multiprocessing?
    for model_path in tqdm(model_paths, desc="Model evaluations"):
        # print(model_path)
        model_filename = parse_model_path(model_path)
        print(model_filename)

        (
            granularity,
            class_id,
            crop_size,
            kernel_size,
            width_factor,
            depth,
        ) = get_parameters(model_filename)

        params = [granularity, class_id, crop_size, kernel_size, width_factor, depth]
        print(params)
        if granularity == "coarse":
            n_labels = 20
            test_dataloader = test_coarse_dataloader
            validation_dataloader = validation_coarse_dataloader
        elif granularity == "fine":
            n_labels = 100
            test_dataloader = test_fine_dataloader
            validation_dataloader = validation_fine_dataloader
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

        # TODO: generate a model profile by calculating the FLOPS
        flops = 999999

        validation_results, validation_accuracy = evaluate(
            validation_dataloader, model, "validation"
        )
        validation_df = pd.DataFrame(validation_results)
        validation_df.to_csv(
            path_or_buf=str(
                evaluations_path / f"validation_eval__{model_filename}.csv"
            ),
            index_label="index",
        )

        profile_ = Profile(*(params + [validation_accuracy, flops]))
        profile_df = pd.DataFrame([profile_], columns=profile_fields)
        profile_df.to_csv(profiles_path, mode="a", header=False)

        test_results, _ = evaluate(test_dataloader, model, "test")
        test_df = pd.DataFrame(test_results)
        test_df.to_csv(
            path_or_buf=str(evaluations_path / f"test_eval__{model_filename}.csv"),
            index_label="index",
        )
        # model_results[model_filename] = {}
        # model_results[model_filename]["validation"] = validation_results
        # model_results[model_filename]["test"] = test_results

    # return model_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gpu", default=6, type=int, help="Index of GPU to use",
    )
    parser.add_argument(
        "--limit", default=5, type=int, help="Limit amount for models to evaluate",
    )
    args = parser.parse_args()
    print("Getting model results")
    main(args)
    # model_results = main(args)
    # pickle_path = str(project_path / "model_results.pkl")
    # print(f"Pickling results to file: {pickle_path}")
    # pickle.dump(model_results, open(pickle_path, "wb"))
