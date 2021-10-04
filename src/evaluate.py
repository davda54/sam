# load the CIFAR test set and validation set
# take in a model artifact
# apply the data transformation (crop size) appropriate to the model
# evaluate the model on the image
# save the results to disk

from pathlib import Path
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os
import argparse
from model.wide_res_net import WideResNet


def get_project_root() -> Path:
    return Path(__file__).parent.parent


root_dir = get_project_root()
dataset_dir = root_dir / "datasets"
dataset_dir.mkdir(parents=True, exist_ok=True)


def cifar100_stats():
    _data_set = torchvision.datasets.CIFAR100(
        root=str(dataset_dir),
        train=True,
        download=True,
        transform=transforms.ToTensor(),
    )

    _data_tensors = torch.cat([d[0] for d in DataLoader(_data_set)])
    mean, std = _data_tensors.mean(dim=[0, 2, 3]), _data_tensors.std(dim=[0, 2, 3])
    return mean, std


def find_model_files(path=(root_dir / "models")):
    model_files = []
    for root, directories, files in os.walk(path):
        for file in files:
            if file.startswith("model_") and file.endswith(".pt"):
                model_files.append(os.path.join(root, file))
    return model_files


def evaluate(dataloader, model, device):
    # idxs = [f"{dataloader.meta.type}ID{i}" for i in range(len(dataloader.targets))]
    model.eval()
    # epoch_loss = 0.0
    # epoch_correct = 0.0
    # epoch_count = 0.0
    output = []
    with torch.no_grad():
        for idx, batch in enumerate(dataloader):
            print(idx)
            inputs, targets = (b.to(device) for b in batch)
            predictions = model(inputs)
            correct = torch.argmax(predictions, 1) == targets
            output.append((targets, predictions, correct))
    return output


# g = evaluate(dataloader, model, device)


def get_test_dataloader():
    mean, std = cifar100_stats()
    test_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean, std)]
    )
    test_dataset = torchvision.datasets.CIFAR100(
        root=str(dataset_dir), train=False, download=False, transform=test_transform,
    )

    test_dataset.meta["type"] = "test"

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1, shuffle=False, num_workers=2,
    )

    return test_dataloader


def get_validation_dataloader():
    validation_dataset = torch.load(
        dataset_dir / "validation" / "validation_dataset.pt"
    )
    validation_dataset.meta["type"] = "validation"
    validation_dataloader = torch.utils.data.DataLoader(
        validation_dataset, batch_size=1, shuffle=False, num_workers=2,
    )
    return validation_dataloader


def main(args):
    """
    from the training set, get 100 images for each superclass
    export as validation set
    """
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    validation_dataloader = get_validation_dataloader()
    test_dataloader = get_test_dataloader()

    model_filepaths = find_model_files()
    for model_fp in model_filepaths:
        # # determine crop size from model_fp
        # # define the crop size transform and then make a dataset obj w/ it in it
        # for element in model_fp.split("_"):
        #     if 'crop' in element:
        #         crop_size = int(element.replace('crop', ''))
        # ts, vs = copy(test_set), copy(validation_set)

        # TODO: Parse out the crop/kernel/width/depth from the model's filepath
        # TODO: Init the model object using the parsed values
        # TODO: Parse out crop or fine from model fp and assign n_labels
        if "coarse":
            n_labels = 20
        else:
            n_labels = 100
        model = WideResNet(
            kernel_size=args.kernel_size,
            width_factor=args.width_factor,
            depth=args.depth,
            dropout=0.0,
            in_channels=3,
            labels=n_labels,
        )
        model_state_dict = torch.load(model_fp, map_location=f"cuda:{args.gpu}")[
            "model_state_dict"
        ]
        model.load_state_dict(model_state_dict)
        model.eval()
        evaluate(model, validation_dataloader, device)
        # evaluate(model, test_dataloader, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gpu", default=6, type=int, help="Index of GPU to use",
    )
    args = parser.parse_args()
    main(args)
