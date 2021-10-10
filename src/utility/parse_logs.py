import os
from pathlib import Path
from typing import List

import pandas as pd
from cifar_utils import coarse_classes


def get_project_root() -> Path:
    return Path(__file__).parent.parent.parent


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
    extension = "." + name.split(".")[-1]
    if param not in ["crop", "kernel", "width", "depth"]:
        raise ValueError("invalid parameter input")
    for element in name.split("_"):
        if param in element:

            return int(element.replace(param, "").replace(extension, ""))


def find_log_files(path=(get_project_root() / "logs" / "model")) -> List[str]:
    log_files = []
    for root, directories, files in os.walk(path):
        for file in files:
            if file.endswith(".log"):
                log_files.append(os.path.join(root, file))
    return log_files


def record_best_epochs(log_files: str):
    search_str = f"achieved a new lowest_loss"
    results = {}
    for fp in log_files:
        try:
            log = search_logfile(fp, search_str)
            name = fp.split("\\")[-1].replace(".log", "")
            epoch = log[0].split(" ")[1]
            loss = log[0].split(" ")[7].rstrip(".")
            test_accuracy = log[1].strip().split("|")[-2].strip()
            results[name] = {}
            results[name]["epoch"] = epoch
            results[name]["loss"] = loss
            results[name]["test_accuracy"] = test_accuracy
        except TypeError as e:
            print(f"TypeError in {fp}: {e}")
    return results


def beautify_df(input_df: pd.DataFrame) -> pd.DataFrame:
    df = input_df.copy()
    df["granularity"] = df.index.map(get_granularity)
    df["superclass"] = df.index.map(get_superclass)
    df["crop"] = df.index.to_series().apply(lambda x: int(get_parameter(x, "crop")))
    df["kernel"] = df.index.to_series().apply(lambda x: int(get_parameter(x, "kernel")))
    df["width"] = df.index.to_series().apply(lambda x: int(get_parameter(x, "width")))
    df["depth"] = df.index.to_series().apply(lambda x: int(get_parameter(x, "depth")))
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
    return df


if __name__ == "__main__":
    log_files = find_log_files()
    best_epochs = record_best_epochs(log_files)
    df = pd.DataFrame(best_epochs).T
    df = beautify_df(df)
    output_path = get_project_root() / "log_results.xlsx"
    print(f"Saving to {output_path}")
    df.to_excel(output_path, columns=df.columns, sheet_name="Log Results")
