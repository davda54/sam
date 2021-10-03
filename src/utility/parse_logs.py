from pathlib import Path
import os
import pandas as pd
from cifar_utils import coarse_classes


def search_logfile(file_path: str, search_str: str) -> None:
    # opening a text file
    file = open(file_path, "r")
    lines = file.readlines()
    lines = list(reversed([line.rstrip() for line in lines]))
    # Loop through the file line by line
    for index, line in enumerate(lines):
        # checking string is present in line or not
        if search_str in line:
            output = lines[index : index + 2]
            return output
    # closing text file
    file.close()


log_files = []
# from utility.misc_utils import get_project_root
# path = get_project_root() / "logs" / "model"
path = Path.cwd() / "logs" / "model"
for root, directories, files in os.walk(path):
    for file in files:
        if file.endswith(".log"):
            log_files.append(os.path.join(root, file))

results = {}
search_str = f"achieved a new lowest_loss"
for file_path in log_files:
    try:
        log_result = search_logfile(file_path, search_str)
        name = file_path.split("\\")[-1].replace(".log", "")
        epoch = log_result[0].split(" ")[1]
        loss = log_result[0].split(" ")[7].rstrip(".")
        test_accuracy = log_result[1].strip().split("|")[-2].strip()
        results[name] = {}
        results[name]["epoch"] = epoch
        results[name]["loss"] = loss
        results[name]["test_accuracy"] = test_accuracy
    except TypeError:
        print(f"TypeError in {file_path}")

df = pd.DataFrame(results).T


def get_granularity(name: str) -> str:
    if "coarse" in name:
        return "coarse"
    elif "fine" in name:
        return "fine"
    else:
        raise ValueError("granularity not found")


def get_superclass(name: str) -> str:
    for cls in coarse_classes:
        if cls in name:
            return cls
    return "all"


def get_parameter(name: str, param: str) -> int:
    if param not in ["crop", "kernel", "width", "depth"]:
        raise ValueError("invalid parameter input")
    for element in name.split("_"):
        if param in element:
            return int(element.replace(param, ""))


df["granularity"] = df.index.map(get_granularity)
df["superclass"] = df.index.map(get_superclass)
df["crop"] = df.index.to_series().apply(lambda x: get_parameter(x, "crop"))
df["kernel"] = df.index.to_series().apply(lambda x: get_parameter(x, "kernel"))
df["width"] = df.index.to_series().apply(lambda x: get_parameter(x, "width"))
df["depth"] = df.index.to_series().apply(lambda x: get_parameter(x, "depth"))

df = df.reindex(
    columns=[
        "granularity",
        "superclass",
        "crop",
        "kernel",
        "width",
        "depth",
        "epoch",
        "loss",
        "test_accuracy",
    ]
)
df.to_excel(
    Path.cwd() / "log_results.xlsx", columns=df.columns, sheet_name="Log Results"
)
