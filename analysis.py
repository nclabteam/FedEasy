import json
import os

import pandas as pd


def format_value(mean, std):
    return f"{mean * 100:.3f}±{std * 100:.3f}"


if __name__ == "__main__":
    paths = [os.path.join("out", "2025-02-27"), os.path.join("out", "2025-02-28")]

    groups = {
        "strategy": [],
        "dataset": [],
        "dirichlet_alpha": [],
        "path": [],
    }

    # Collect data paths from directories
    for path in paths:
        for root, dirs, files in os.walk(path):
            for dir in dirs:
                subpath = os.path.join(root, dir)
                config_path = os.path.join(subpath, "config.json")
                if not os.path.exists(config_path):
                    continue
                with open(config_path) as f:
                    config = json.load(f)
                    groups["strategy"].append(config["server"]["strategy"])
                    groups["dataset"].append(config["common"]["dataset"])
                    groups["dirichlet_alpha"].append(
                        config["common"]["dirichlet_alpha"]
                    )
                    groups["path"].append(
                        os.path.join(
                            subpath,
                            f'{config["server"]["strategy"]}_{config["common"]["dataset"]}_{config["common"]["data_type"]}_{config["client"]["batch_size"]}_{config["client"]["lr"]}_{config["client"]["epochs"]}.csv',
                        )
                    )

    data = {
        "strategy": [],
        "dataset": [],
        "dirichlet_alpha": [],
        "max_acc": [],
        "max_acc_idx": [],
        "@20": [],
        "@100": [],
        "@300": [],
        "mean-90_100": [],
        "mean-290_300": [],
    }

    # Process grouped data
    for name, gdf in pd.DataFrame(groups).groupby(
        ["strategy", "dataset", "dirichlet_alpha"]
    ):
        dfs = []
        if len(gdf["path"].to_list()) != 3:
            continue
        for csv_path in gdf["path"].to_list():
            dfs.append(pd.read_csv(csv_path))

        dfs = (
            pd.concat(dfs)
            .groupby("round", as_index=False)
            .agg(["mean", "std"])
            .set_index("round")
        )

        # Extract values and standard deviations
        max_acc_idx = dfs["accuracy"]["mean"].idxmax()
        # if name[0] == "FedAvgM":
        #     print(max_acc_idx)
        #     print(dfs["accuracy"]["std"])
        #     break
        max_acc_mean = dfs["accuracy"]["mean"].iloc[max_acc_idx - 1]
        max_acc_std = dfs["accuracy"]["std"].iloc[max_acc_idx - 1]

        data["strategy"].append(name[0])
        data["dataset"].append(name[1])
        data["dirichlet_alpha"].append(name[2])
        data["max_acc"].append(format_value(max_acc_mean, max_acc_std))
        data["max_acc_idx"].append(int(dfs.index[max_acc_idx - 1]))

        # Get accuracy stats at specific rounds
        for key, r in {"@20": 20, "@100": 100, "@300": 300}.items():
            if r in dfs.index:  # Ensures round numbers match
                data[key].append(
                    format_value(
                        dfs["accuracy"]["mean"].loc[r], dfs["accuracy"]["std"].loc[r]
                    )
                )
            else:
                data[key].append("N/A")

        # Get mean and std over ranges
        for key, r in {"mean-90_100": (90, 100), "mean-290_300": (290, 300)}.items():
            filtered = dfs.loc[(dfs.index > r[0]) & (dfs.index <= r[1]), "accuracy"]
            if not filtered.empty:
                data[key].append(
                    format_value(filtered["mean"].mean(), filtered["std"].mean())
                )
            else:
                data[key].append("N/A")

    data = pd.DataFrame(data)
    data.to_csv("results.csv")
    for name, gdf in data.groupby(["dirichlet_alpha"]):
        print(gdf.sort_values(by="mean-290_300", ascending=False))
        print("=" * 100)
