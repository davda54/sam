import argparse
import os
from pathlib import Path

import GPUtil
import numpy as np
import torch
import torchvision
from torchvision.transforms import transforms

from model.smooth_cross_entropy import smooth_crossentropy
from model.wide_res_net import WideResNet
from sam import SAM
from utility.bypass_bn import disable_running_stats, enable_running_stats
from utility.cifar_utils import load_dataset
from utility.initialize import initialize
from utility.log import Log
from utility.step_lr import StepLR


def get_project_root() -> Path:
    return Path(__file__).parent.parent


# import sys
#
# sys.path.append(get_project_root)

# TODO: Collapse crop/padding functions down to a single one
def set_crop_size(dataloader, crop_size: int):
    """
    takes in a dataloader containing dataset CIFAR100Indexed and sets size of the RandomCrop
    """
    for i, t in enumerate(dataloader.dataset.transforms.transform.transforms):
        if type(t) == torchvision.transforms.transforms.RandomCrop:
            dataloader.dataset.transforms.transform.transforms[i].size = (
                crop_size,
                crop_size,
            )


def set_padding_amount(dataloader, padding_amount: int):
    """
    takes in a dataloader containing dataset CIFAR100Indexed and sets amount of padding for the RandomCrop
    """
    for i, t in enumerate(dataloader.dataset.transforms.transform.transforms):
        if type(t) == transforms.RandomCrop:
            dataloader.dataset.transforms.transform.transforms[
                i
            ].padding = padding_amount


def determine_params(crop_size: int) -> (int, int):
    """
    # !!! NOTE KERNEL AND PADDING SIZE DEPENDS ON CROP SIZE !!!
    # crop 32 = kernel 8
    # crop 24 = kernel 6
    # crop 16 = kernel 4
    # crop 8 = kernel 2
    :param crop_size:
    :return: kernel_size, padding_amount
    """
    if crop_size == 32:
        return 8, 4
    elif crop_size == 24:
        return 6, 4 # could be 3 instead
    elif crop_size == 16:
        return 4, 4 # could be 2 instead
    elif crop_size == 8:
        return 2, 4 # could be 1 instead
    else:
        raise ValueError("invalid crop size")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gpu", default=-1, type=int, help="Index value for the GPU to use",
    )
    parser.add_argument("--fine_classes", dest="use_fine_classes", action="store_true")
    parser.add_argument(
        "--coarse_classes", dest="use_fine_classes", action="store_false",
    )
    parser.set_defaults(use_fine_classes=True)
    parser.add_argument(
        "--superclass", default="all", type=str, help="Superclass we want to use",
    )
    parser.add_argument(
        "--adaptive",
        default=True,
        type=bool,
        help="True if you want to use the Adaptive SAM.",
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
    parser.add_argument("--depth", default=16, type=int, help="Number of layers.")
    parser.add_argument(
        "--width_factor",
        default=8,
        type=int,
        help="How many times wider compared to normal ResNet.",
    )
    parser.add_argument("--dropout", default=0.0, type=float, help="Dropout rate.")
    parser.add_argument(
        "--epochs", default=200, type=int, help="Total number of epochs."
    )
    parser.add_argument(
        "--label_smoothing",
        default=0.1,
        type=float,
        help="Use 0.0 for no label smoothing.",
    )
    parser.add_argument(
        "--learning_rate",
        default=0.1,
        type=float,
        help="Base learning rate at the start of the training.",
    )
    parser.add_argument("--momentum", default=0.9, type=float, help="SGD Momentum.")
    parser.add_argument(
        "--threads", default=2, type=int, help="Number of CPU threads for dataloaders."
    )
    parser.add_argument("--rho", default=2.0, type=int, help="Rho parameter for SAM.")
    parser.add_argument(
        "--weight_decay", default=0.0005, type=float, help="L2 weight decay."
    )
    args = parser.parse_args()

    print(args)

    if args.use_fine_classes:
        args.granularity = "fine"
        if not args.superclass:
            ValueError("Must provide superclass when training with fine labels")
        superclass = str(args.superclass)
    else:
        args.granularity = "coarse"
        if not args.superclass:
            superclass = "all"

    # Determine the kernel size and padding amount from the crop size argument
    kernel_size, padding_amount = determine_params(args.crop_size)

    initialize(args, seed=42)

    # TODO: GPU Selection is not currently implemented correctly -- it hits memory issues
    if args.gpu == -1:
        # Set CUDA_DEVICE_ORDER so the IDs assigned by CUDA match those from nvidia-smi
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        # Get the first available GPU
        DEVICE_ID_LIST = GPUtil.getFirstAvailable()
        DEVICE_ID = DEVICE_ID_LIST[0]  # grab first element from list
        device = torch.device(
            f"cuda:{DEVICE_ID}" if torch.cuda.is_available() else "cpu"
        )
        # # Set CUDA_VISIBLE_DEVICES to mask out all other GPUs than the first available device id
        # os.environ["CUDA_VISIBLE_DEVICES"] = str(DEVICE_ID)
        # # Since all other GPUs are masked out, the first available GPU will now be identified as GPU:0
        # print('Device ID (unmasked): ' + str(DEVICE_ID))
        # print('Device ID (masked): ' + str(0))
    else:
        device = torch.device(
            f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
        )

    train_set = torch.utils.data.DataLoader(
        load_dataset("train", args),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.threads,
    )
    # Set the crop size and padding amount
    set_crop_size(train_set, kernel_size)
    set_padding_amount(train_set, padding_amount)

    test_set = torch.utils.data.DataLoader(
        load_dataset("test", args),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.threads,
    )
    # Set the crop size (no padding on training)
    set_crop_size(test_set, kernel_size)

    fp = (
        get_project_root()
        / "models"
        / args.granularity
        / args.superclass
        / f"crop{str(args.crop_size)}_kernel{str(kernel_size)}_padding{str(padding_amount)}"
        / f"width{str(args.width_factor)}"
        / f"depth{str(args.depth)}"
        / f"model_{args.granularity}_{args.superclass}_crop{args.crop_size}_kernel{str(kernel_size)}_width{args.width_factor}_depth{args.depth}.pt"
    )

    fp.parent.mkdir(parents=True, exist_ok=True)

    log = Log(log_each=10)

    if args.use_fine_classes:
        n_labels = 100
    else:
        n_labels = 20

    model = WideResNet(
        depth=args.depth,
        width_factor=args.width_factor,
        dropout=args.dropout,
        kernel_size=kernel_size,
        in_channels=3,
        labels=n_labels,
    ).to(device)

    base_optimizer = torch.optim.SGD
    optimizer = SAM(
        model.parameters(),
        base_optimizer,
        rho=args.rho,
        adaptive=args.adaptive,
        lr=args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    scheduler = StepLR(optimizer, args.learning_rate, args.epochs)

    lowest_loss = (
        np.inf
    )  # Starts at negative infinity to ensure we capture first result regardless
    for epoch in range(args.epochs):
        model.train()
        log.train(len_dataset=len(train_set))

        for batch in train_set:
            inputs, targets = (b.to(device) for b in batch)

            # first forward-backward step
            enable_running_stats(model)
            predictions, _ = model(inputs)  # Ignore the embedding output
            loss = smooth_crossentropy(predictions, targets)
            loss.mean().backward()
            optimizer.first_step(zero_grad=True)

            # second forward-backward step
            disable_running_stats(model)
            predictions_2nd, _ = model(inputs)  # Ignore the embedding output
            smooth_crossentropy(predictions_2nd, targets).mean().backward()
            optimizer.second_step(zero_grad=True)

            with torch.no_grad():
                correct = torch.argmax(predictions.data, 1) == targets
                log(model, loss.cpu(), correct.cpu(), scheduler.lr())
                scheduler(epoch)

        model.eval()
        log.eval(len_dataset=len(test_set))
        epoch_loss, epoch_correct, epoch_count = 0.0, 0.0, 0.0
        with torch.no_grad():
            for batch in test_set:
                inputs, targets = (b.to(device) for b in batch)

                predictions, _ = model(inputs)  # Ignore the embedding output
                loss = smooth_crossentropy(predictions, targets)
                batch_loss = loss.sum().item()
                epoch_loss += batch_loss
                correct = torch.argmax(predictions, 1) == targets
                batch_correct = correct.sum().item()
                epoch_correct += batch_correct
                epoch_count += len(targets)
                log(model, loss.cpu(), correct.cpu())

        log.flush()

        if epoch_loss < lowest_loss:
            print(
                f"Epoch {epoch} achieved a new lowest_loss of {epoch_loss}. Saving model to disk."
            )
            lowest_loss = epoch_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": epoch_loss,
                    "correct": epoch_correct,
                    "size": epoch_count,
                    "accuracy": epoch_correct / epoch_count,
                },
                str(fp),
            )
