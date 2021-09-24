import argparse
import torch

from model.wide_res_net import WideResNet
from model.smooth_cross_entropy import smooth_crossentropy
from data.cifar10 import CifarTen
from data.cifar100 import CifarHundred
from utility.log import Log
from utility.initialize import initialize
from utility.step_lr import StepLR
from utility.bypass_bn import enable_running_stats, disable_running_stats

import sys

from pathlib import Path

sys.path.append("..")
from sam import SAM


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train100",
        default=True,
        type=bool,
        help="True to use CIFAR100, False for CIFAR10.",
    )
    parser.add_argument(
        "--coarse_labels",
        default=True,
        type=bool,
        help="True to use CIFAR100 coarse granularity of superclasses (20), False for fine class granularity (100).",
    )
    parser.add_argument(
        "--adaptive",
        default=True,
        type=bool,
        help="True if you want to use the Adaptive SAM.",
    )
    parser.add_argument(
        "--batch_size",
        default=128,
        type=int,
        help="Batch size used in the training and validation loop.",
    )
    parser.add_argument("--depth", default=16, type=int, help="Number of layers.")
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
    parser.add_argument(
        "--width_factor",
        default=8,
        type=int,
        help="How many times wider compared to normal ResNet.",
    )
    args = parser.parse_args()

    print(args)

    initialize(args, seed=42)
    device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")

    if args.train100:
        dataset = CifarHundred(args.batch_size, args.threads)
    else:
        dataset = CifarTen(args.batch_size, args.threads)

    log = Log(log_each=10)

    if not args.train100:
        model = WideResNet(
            args.depth, args.width_factor, args.dropout, in_channels=3, labels=10
        ).to(device)
    elif args.train100 and args.coarse_labels:
        model = WideResNet(
            args.depth, args.width_factor, args.dropout, in_channels=3, labels=20
        ).to(device)
    else:
        model = WideResNet(
            args.depth, args.width_factor, args.dropout, in_channels=3, labels=100
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

    for epoch in range(args.epochs):
        model.train()
        log.train(len_dataset=len(dataset.train))

        for batch in dataset.train:
            inputs, targets = (b.to(device) for b in batch)

            # first forward-backward step
            enable_running_stats(model)
            predictions = model(inputs)
            loss = smooth_crossentropy(predictions, targets)
            loss.mean().backward()
            optimizer.first_step(zero_grad=True)

            # second forward-backward step
            disable_running_stats(model)
            smooth_crossentropy(model(inputs), targets).mean().backward()
            optimizer.second_step(zero_grad=True)

            with torch.no_grad():
                correct = torch.argmax(predictions.data, 1) == targets
                log(model, loss.cpu(), correct.cpu(), scheduler.lr())
                scheduler(epoch)

        model.eval()
        log.eval(len_dataset=len(dataset.test))

        with torch.no_grad():
            for batch in dataset.test:
                inputs, targets = (b.to(device) for b in batch)

                predictions = model(inputs)
                loss = smooth_crossentropy(predictions, targets)
                correct = torch.argmax(predictions, 1) == targets
                log(model, loss.cpu(), correct.cpu())

    model_filename = str(Path.cwd()/'output'/f"model_coarse{args.coarse_labels}_width{args.width_factor}_depth{args.depth}")

    torch.save(obj=model.state_dict(), f=model_filename)

    log.flush()
