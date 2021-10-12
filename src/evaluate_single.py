import argparse
import csv
import os
from pathlib import Path

import pandas as pd
import torch
import torchvision
import torchvision.transforms as transforms
from ptflops import get_model_complexity_info
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from model.wide_res_net import WideResNet
from evaluate import set_crop_size
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

predictions_path = evaluations_path / "predictions"
predictions_path.mkdir(exist_ok=True, parents=True)

embeddings_path = evaluations_path / "embeddings"
embeddings_path.mkdir(exist_ok=True, parents=True)

Result = namedtuple("Result", ["idx", "prediction", "target", "correct", "outputs"])
profile_fields = [
    "granularity",
    "superclass",
    "crop_size",
    "kernel_size",
    "width_factor",
    "depth",
    "accuracy",
    "macs",
    "flops",
    "params",
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
    if "_all_" in filename:  # if coarse just removes the superclass placeholder
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


def evaluate(dataloader, model, device, dataset_type: str):
    """

    :param dataloader: Dataloader containing CIFAR100Indexed
    :param model: WideResNet model object
    :param device: CUDA device
    :param dataset_type: Text string with the data split (train, test, validate, etc)
    :return:
    """
    model_results = []
    model_embeddings = {}
    # total_loss = 0.0
    total_correct = 0.0
    count = 0.0
    with torch.no_grad():
        for inputs, targets, idxs in tqdm(
            dataloader, desc=f"Evaluating {dataset_type} data", leave=False
        ):
            inputs, targets = inputs.to(device), targets.to(device)
            count += len(inputs)
            outputs, embeds = model(inputs)
            # TODO: Put `embeddings` in dict with each embedding keyed to the image index and pickle the dictionary
            # total_loss += smooth_crossentropy(outputs, targets)
            predictions = torch.argmax(outputs, 1)
            correct = predictions == targets
            total_correct += correct.sum().item()

            # Data munging for predictions
            predict_zip = zip(
                idxs,
                zip(*(predictions.cpu(), targets.cpu(), correct.cpu(), outputs.cpu())),
            )
            for idx, data in predict_zip:
                result_ = [idx.tolist()] + [d.tolist() for d in data]
                model_results.append(Result(*result_))

            # Data munging for embeddings
            embeds_zip = zip(idxs, embeds.cpu())
            for idx, embed in embeds_zip:
                model_embeddings[idx.tolist()] = embed.tolist()
    accuracy = total_correct / count
    return model_results, accuracy, model_embeddings


def split_outputs_column(df: pd.DataFrame, n_outputs: int):
    """
    Split the array elements in the `outputs` column into individual columns in pandas
    :param df: DataFrame containing an outputs column
    :return: DataFrame with the outputs column split by element
    """
    # new df from the column of lists
    outputs_df = pd.DataFrame(
        df["outputs"].tolist(), columns=[f"output{i}" for i in range(n_outputs)],
    )
    # attach output columns back to df
    df = pd.concat([df, outputs_df], axis=1)
    df = df.drop("outputs", axis=1)  # drop the original outputs column
    return df


def get_test_dataloader(coarse=False):
    mean, std = cifar100_stats(root=str(dataset_path))
    test_transform = transforms.Compose(
        [
            transforms.RandomCrop(size=32),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
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
        test_dataset, batch_size=1024, shuffle=False, num_workers=2,
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
        validation_dataset, batch_size=1024, shuffle=False, num_workers=2,
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
    validation_fine_dataloader = get_validation_dataloader(coarse=False)
    validation_coarse_dataloader = get_validation_dataloader(coarse=True)

    # TODO: Instead of passing in an entire model path, pass in the name of the model and find it in the directory
    # model_paths = find_model_files()
    # model_paths = model_paths[: _args.limit]

    profiles_path = evaluations_path / "model_profiles.csv"
    with open(profiles_path, "w", encoding="UTF8") as f:
        writer = csv.writer(f)
        writer.writerow(profile_fields)

    model_path = _args.model_path
    print(model_path)
    model_filename = parse_model_path(str(model_path))

    print(model_filename)

    (
        granularity,
        class_id,
        crop_size,
        kernel_size,
        width_factor,
        depth,
    ) = get_parameters(model_filename)

    model_info = [
        granularity,
        class_id,
        crop_size,
        kernel_size,
        width_factor,
        depth,
    ]
    print(model_info)

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

    # Sets the crop size on the RandomCrop transform to fit the model
    set_crop_size(test_dataloader, crop_size)
    set_crop_size(validation_dataloader, crop_size)

    # TODO: [OPTIONAL] Set the dataloader's batch size based on the crop size to increase evaluation speed

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
    model.cuda(device)
    model.eval()

    test_results, test_accuracy, test_embeddings = evaluate(
        test_dataloader, model, device, "test"
    )
    test_df = pd.DataFrame(test_results)
    test_df = split_outputs_column(test_df, n_labels)

    test_df.to_csv(
        path_or_buf=str(predictions_path / f"test_eval__{model_filename}.csv"),
        index=False,
    )
    import pickle

    test_embeds_pkl = open(
        str(embeddings_path / f"test_embeds__{model_filename}.pkl"), "wb",
    )
    pickle.dump(test_embeddings, test_embeds_pkl)
    test_embeds_pkl.close()
    macs, params = get_model_complexity_info(
        model,
        (3, crop_size, crop_size),
        as_strings=True,
        print_per_layer_stat=False,
        verbose=False,
    )
    flops = f"{2*float(macs.split(' ')[0])} GFLOPs"

    validation_results, validation_accuracy, validation_embeddings = evaluate(
        validation_dataloader, model, device, "validation"
    )
    validation_embeds_pkl = open(
        str(embeddings_path / f"validation_embeds__{model_filename}.pkl"), "wb",
    )
    pickle.dump(validation_embeddings, validation_embeds_pkl)
    validation_embeds_pkl.close()

    validation_df = pd.DataFrame(validation_results)
    validation_df = split_outputs_column(validation_df, n_labels)
    validation_df.to_csv(
        path_or_buf=str(
            predictions_path / f"validation_eval__{model_filename}.csv"
        ),
        index=False,
    )

    profile_ = Profile(*(model_info + [validation_accuracy, macs, flops, params]))
    profile_df = pd.DataFrame([profile_], columns=profile_fields)
    profile_df.to_csv(profiles_path, mode="a", header=False, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gpu", default=3, type=int, help="Index of GPU to use",
    )
    parser.add_argument(
        "--model_path",
        default=None,
        type=str,
        help="Path to the model we want to evaluate",
    )
    args = parser.parse_args()
    print("Getting model results")
    main(args)
