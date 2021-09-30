import numpy as np
import torch
import torchvision
from pathlib import Path

def get_project_root() -> Path:
    return Path(__file__).parent.parent.parent

fine_classes = (
    "apple",
    "aquarium_fish",
    "baby",
    "bear",
    "beaver",
    "bed",
    "bee",
    "beetle",
    "bicycle",
    "bottle",
    "bowl",
    "boy",
    "bridge",
    "bus",
    "butterfly",
    "camel",
    "can",
    "castle",
    "caterpillar",
    "cattle",
    "chair",
    "chimpanzee",
    "clock",
    "cloud",
    "cockroach",
    "couch",
    "crab",
    "crocodile",
    "cup",
    "dinosaur",
    "dolphin",
    "elephant",
    "flatfish",
    "forest",
    "fox",
    "girl",
    "hamster",
    "house",
    "kangaroo",
    "keyboard",
    "lamp",
    "lawn_mower",
    "leopard",
    "lion",
    "lizard",
    "lobster",
    "man",
    "maple_tree",
    "motorcycle",
    "mountain",
    "mouse",
    "mushroom",
    "oak_tree",
    "orange",
    "orchid",
    "otter",
    "palm_tree",
    "pear",
    "pickup_truck",
    "pine_tree",
    "plain",
    "plate",
    "poppy",
    "porcupine",
    "possum",
    "rabbit",
    "raccoon",
    "ray",
    "road",
    "rocket",
    "rose",
    "sea",
    "seal",
    "shark",
    "shrew",
    "skunk",
    "skyscraper",
    "snail",
    "snake",
    "spider",
    "squirrel",
    "streetcar",
    "sunflower",
    "sweet_pepper",
    "table",
    "tank",
    "telephone",
    "television",
    "tiger",
    "tractor",
    "train",
    "trout",
    "tulip",
    "turtle",
    "wardrobe",
    "whale",
    "willow_tree",
    "wolf",
    "woman",
    "worm",
)

coarse_classes = (
    "aquatic_mammals",
    "fish",
    "flowers",
    "food_containers",
    "fruit_and_vegetables",
    "household_electrical_devices",
    "household_furniture",
    "insects",
    "large_carnivores",
    "large_man - made_outdoor_things",
    "large_natural_outdoor_scenes",
    "large_omnivores_and_herbivores",
    "medium_mammals",
    "non - insect_invertebrates",
    "people",
    "reptiles",
    "small_mammals",
    "trees",
    "vehicles_1",
    "vehicles_2",
)

coarse_idxs = dict([L[::-1] for L in enumerate(coarse_classes)])

coarse_classes_key = np.array(
    [
        4,
        1,
        14,
        8,
        0,
        6,
        7,
        7,
        18,
        3,
        3,
        14,
        9,
        18,
        7,
        11,
        3,
        9,
        7,
        11,
        6,
        11,
        5,
        10,
        7,
        6,
        13,
        15,
        3,
        15,
        0,
        11,
        1,
        10,
        12,
        14,
        16,
        9,
        11,
        5,
        5,
        19,
        8,
        8,
        15,
        13,
        14,
        17,
        18,
        10,
        16,
        4,
        17,
        4,
        2,
        0,
        17,
        4,
        18,
        17,
        10,
        3,
        2,
        12,
        12,
        16,
        12,
        1,
        9,
        19,
        2,
        10,
        0,
        1,
        16,
        12,
        9,
        13,
        15,
        13,
        16,
        19,
        2,
        4,
        6,
        19,
        5,
        5,
        8,
        19,
        18,
        1,
        2,
        15,
        6,
        0,
        17,
        8,
        14,
        13,
    ]
)

coarse_classes_map = dict(enumerate(coarse_classes_key))


def load_dataset(split: str, _args):
    if split not in ["train", "test"]:
        raise ValueError("split must be 'train' or 'test'")
    fp = (
        get_project_root()
        / "datasets"
        / split
        / _args.granularity
        / _args.superclass
        / f"crop_size{str(_args.crop_size)}"
        / f"dataset_{split}_{_args.granularity}_{_args.superclass}_crop{str(_args.crop_size)}.pt"
    )
    dataset = torch.load(fp)
    return dataset


def save_dataset(data: torchvision.datasets, split: str, _args):
    if split not in ["train", "test"]:
        raise ValueError("split must be 'train' or 'test'")
    fp = (
        get_project_root()
        / "datasets"
        / split
        / _args.granularity
        / _args.superclass
        / f"crop_size{str(_args.crop_size)}"
        / f"dataset_{split}_{_args.granularity}_{_args.superclass}_crop{str(_args.crop_size)}.pt"
    )
    fp.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving: {fp}")
    torch.save(data, fp)
