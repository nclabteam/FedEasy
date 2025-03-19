import argparse
import csv
import json
import os
import random
from datetime import date, datetime
from logging import INFO
from typing import Dict

import flwr as fl
import numpy as np
import pandas as pd
import torch
import yaml
from datasets import Dataset
from datasets.utils.logging import disable_progress_bar
from flwr.common import Scalar
from flwr.common.logger import log
from flwr.common.typing import Scalar
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import DirichletPartitioner, IidPartitioner
from torch.utils.data import DataLoader

import mak
import mak.strategies
from mak.servers.custom_server import ServerSaveData
from mak.servers.fednova_server import FedNovaServer
from mak.servers.scaffold_server import ScaffoldServer
from mak.strategies.fednova_strategy import FedNovaStrategy
from mak.strategies.scaffold_strategy import ScaffoldStrategy
from mak.utils.dataset_info import dataset_info
from mak.utils.general import set_params, test, weighted_average


def get_device_and_resources(config_sim):
    # Check if GPU is available
    device = torch.device(
        "cuda" if torch.cuda.is_available() and config_sim["client"]["gpu"] else "cpu"
    )

    # Assign GPU and CPU resources
    if device.type == "cuda":
        # Assign GPU resources
        num_gpus_total = config_sim["client"]["total_gpus"]
        if num_gpus_total > 0:
            ray_init_args = {
                "num_cpus": config_sim["client"]["total_cpus"],
                "num_gpus": num_gpus_total,
            }
        else:
            ray_init_args = {
                "num_cpus": config_sim["client"]["total_cpus"],
                "num_gpus": 0,
            }
    else:
        # Assign CPU resources
        ray_init_args = {"num_cpus": config_sim["client"]["total_cpus"], "num_gpus": 0}

    # Assign client resources
    client_res = {
        "num_cpus": config_sim["client"]["num_cpus"],
        "num_gpus": config_sim["client"]["num_gpus"] if device.type == "cuda" else 0.0,
    }
    if config_sim["common"]["multi_node"]:
        ray_init_args = {}
        ray_init_args["address"] = "auto"
        ray_init_args["runtime_env"] = {"py_modules": [mak]}
    return device, ray_init_args, client_res


def gen_dir_outfile_server(config):
    # generates the basic directory structure for out data and the header for file
    today = date.today()
    BASE_DIR = "out"
    if not os.path.exists(BASE_DIR):
        os.mkdir(BASE_DIR)

    # create a date wise folder
    if not os.path.exists(os.path.join(BASE_DIR, str(today))):
        os.mkdir(os.path.join(BASE_DIR, str(today)))

    # create saperate folder based on strategy
    if not os.path.exists(
        os.path.join(BASE_DIR, str(today), config["server"]["strategy"])
    ):
        os.mkdir(os.path.join(BASE_DIR, str(today), config["server"]["strategy"]))

    # create saperate folder based on data distribution type
    if not os.path.exists(
        os.path.join(
            BASE_DIR,
            str(today),
            config["server"]["strategy"],
            config["common"]["data_type"],
        )
    ):
        os.mkdir(
            os.path.join(
                BASE_DIR,
                str(today),
                config["server"]["strategy"],
                config["common"]["data_type"],
            )
        )

    dirs = os.listdir(
        os.path.join(
            BASE_DIR,
            str(today),
            config["server"]["strategy"],
            config["common"]["data_type"],
        )
    )
    final_dir_path = os.path.join(
        BASE_DIR,
        str(today),
        config["server"]["strategy"],
        config["common"]["data_type"],
        str(len(dirs)),
    )

    if not os.path.exists(final_dir_path):
        os.mkdir(final_dir_path)
    if not os.path.exists(os.path.join(final_dir_path, "clients")):
        os.mkdir(os.path.join(final_dir_path, "clients"))
    # models_dir = os.path.join(final_dir_path,'models')
    now = datetime.now()
    current_time = now.strftime("%H-%M-%S")
    # save all confugration file as json file
    json_file_name = f"config.json"
    with open(os.path.join(final_dir_path, json_file_name), "w") as fp:
        json.dump(config, fp, indent=4)
    dataset_str = config["common"]["dataset"].replace("/", "_")
    file_name = f"{config['server']['strategy']}_{dataset_str}_{config['common']['data_type']}_{config['client']['batch_size']}_{config['client']['lr']}_{config['client']['epochs']}"
    file_name = f"{file_name}.csv"
    out_file_path = os.path.join(final_dir_path, file_name)
    # create empty server history file
    if not os.path.exists(out_file_path):
        with open(out_file_path, "w", encoding="UTF8") as f:
            # create the csv writer
            header = ["round", "accuracy", "loss", "time"]
            writer = csv.writer(f)
            writer.writerow(header)
            f.close()
    return out_file_path, final_dir_path


def get_partitioner(config_sim):
    num_clients = config_sim["server"]["num_clients"]
    if config_sim["common"]["data_type"] == "dirichlet_niid":
        # alpha value
        dirchlet_alpha = config_sim["common"]["dirichlet_alpha"]
        # dataset
        dataset_name = config_sim["common"]["dataset"]
        # dataset's label column
        label = dataset_info[dataset_name]["output_column"]
        partitioner = DirichletPartitioner(
            num_partitions=num_clients,
            partition_by=label,
            alpha=dirchlet_alpha,
            min_partition_size=2,
            self_balancing=True,
        )
    else:
        partitioner = IidPartitioner(num_partitions=num_clients)
    # return train data
    return {"train": partitioner}


def get_dataset(config_sim):
    partitioner = get_partitioner(config_sim=config_sim)

    dataset_name = config_sim["common"]["dataset"]
    if dataset_name not in dataset_info.keys():
        raise Exception(f"Dataset name should be among : {list(dataset_info.keys())}")
    else:
        fds = FederatedDataset(dataset=dataset_name, partitioners=partitioner)
        # get test column name
        test_set = dataset_info[dataset_name]["test_set"]
        centralized_testset = fds.load_split(test_set)

        return fds, centralized_testset


def get_model(config, shape):
    model_name = config["common"]["model"]
    # get num_classes
    dataset_name = config["common"]["dataset"]
    num_classes = dataset_info[dataset_name]["num_classes"]
    # get model
    model = getattr(__import__("mak.models", fromlist=[model_name]), model_name)(
        num_classes=num_classes, input_shape=shape
    )
    return model


def get_evaluate_fn(
    centralized_testset: Dataset,
    config_sim,
    device,
    save_model_dir,
    metrics_file,
    apply_transforms_test,
):
    """Return an evaluation function for centralized evaluation."""
    dataset_name = config_sim["common"]["dataset"]
    shape = dataset_info[dataset_name]["input_shape"]

    def evaluate(
        server_round: int, parameters: fl.common.NDArrays, config: Dict[str, Scalar]
    ):
        model = get_model(config=config_sim, shape=shape)
        set_params(model, parameters)
        model.to(device)

        # Apply transform to dataset
        testset = centralized_testset.with_transform(apply_transforms_test)

        # Disable tqdm for dataset preprocessing
        disable_progress_bar()

        testloader = DataLoader(testset, batch_size=32)
        loss, accuracy = test(model, testloader, device=device)
        metrics_df = pd.read_csv(metrics_file)
        if metrics_df["loss"].min() > loss:
            log(
                INFO,
                f" =>>>>> Min Loss improved from {metrics_df['loss'].min()} to : {loss} Saving best model having accuracy : {accuracy}",
            )
            torch.save(
                model.state_dict(), os.path.join(save_model_dir, "saved_best_model.pth")
            )

        if server_round == config_sim["server"]["num_rounds"]:
            torch.save(
                model.state_dict(),
                os.path.join(save_model_dir, "saved_model_final.pth"),
            )
        return loss, {"accuracy": accuracy}

    return evaluate


def save_simulation_history(hist: fl.server.history.History, path):
    losses_distributed = hist.losses_distributed
    losses_centralized = hist.losses_centralized
    metrics_distributed_fit = hist.metrics_distributed_fit
    metrics_distributed = hist.metrics_distributed
    metrics_centralized = hist.metrics_centralized

    rounds = []
    losses_centralized_dict = {}
    losses_distributed_dict = {}
    accuracy_distributed_dict = {}
    accuracy_centralized_dict = {}

    for loss in losses_centralized:
        c_rnd = loss[0]
        rounds.append(c_rnd)
        losses_centralized_dict[c_rnd] = loss[1]

    for loss in losses_distributed:
        c_rnd = loss[0]
        losses_distributed_dict[c_rnd] = loss[1]
    if "accuracy" in metrics_distributed.keys():
        for acc in metrics_distributed["accuracy"]:
            c_rnd = acc[0]
            accuracy_distributed_dict[c_rnd] = acc[1]
    if "accuracy" in metrics_centralized.keys():
        for acc in metrics_centralized["accuracy"]:
            c_rnd = acc[0]
            accuracy_centralized_dict[c_rnd] = acc[1]

    if len(metrics_distributed_fit) != 0:
        pass  # TODO  check its implemetation later

    data = {
        "rounds": rounds,
        "losses_centralized": losses_centralized_dict,
        "losses_distributed": losses_distributed_dict,
        "accuracy_distributed": accuracy_distributed_dict,
        "accuracy_centralized": accuracy_centralized_dict,
    }

    # Create an empty DataFrame
    df = pd.DataFrame()

    # Iterate over each key in the data dictionary
    for key in data.keys():
        # If the key is 'rounds', set the 'rounds' column of the DataFrame to the rounds list
        if key == "rounds":
            df["rounds"] = data[key]
        # Otherwise, create a new column in the DataFrame with the key as the column name
        else:
            column_data = []
            # Iterate over each round in the 'rounds' list and add the corresponding value for the current key
            for round_num in data["rounds"]:
                # If the round number does not exist in the current key's dictionary, set the value to None
                if round_num not in data[key]:
                    column_data.append(None)
                else:
                    column_data.append(data[key][round_num])
            df[key] = column_data
    df.to_csv(os.path.join(path))


def get_server(strategy, client_manager, out_file_path, target_acc):
    if isinstance(strategy, ScaffoldStrategy):
        return ScaffoldServer(
            strategy=strategy,
            client_manager=client_manager,
            out_file_path=out_file_path,
            target_acc=target_acc,
        )
    elif isinstance(strategy, FedNovaStrategy):
        return FedNovaServer(
            strategy=strategy,
            client_manager=client_manager,
            out_file_path=out_file_path,
            target_acc=target_acc,
        )
    else:
        return ServerSaveData(
            strategy=strategy,
            client_manager=client_manager,
            out_file_path=out_file_path,
            target_acc=target_acc,
        )


def get_strategy(
    config,
    test_data,
    save_model_dir,
    out_file_path,
    device,
    apply_transforms_test,
    size_weights,
):
    STRATEGY = config["server"]["strategy"]
    dataset_name = config["common"]["dataset"]
    shape = dataset_info[dataset_name]["input_shape"]
    model = get_model(config=config, shape=shape)
    MIN_CLIENTS_FIT = config["server"]["min_fit_clients"]
    MIN_CLIENTS_EVAL = 2
    NUM_CLIENTS = config["server"]["num_clients"]
    FRACTION_FIT = config["server"]["fraction_fit"]
    FRACTION_EVAL = config["server"]["fraction_evaluate"]

    kwargs = {
        "FedAvgM": {
            "server_learning_rate": 1.0,
            "server_momentum": 0.2,
        },
        "FedAdam": {
            "eta": 1e-1,
            "eta_l": 1e-1,
            "beta_1": 0.9,
            "beta_2": 0.99,
            "tau": 1e-9,
        },
        "FedOpt": {
            "eta": 1e-1,
            "eta_l": 1e-1,
            "beta_1": 0.0,
            "beta_2": 0.0,
            "tau": 1e-9,
        },
        "FedProx": {
            "proximal_mu": config["fedprox"]["proximal_mu"],
        },
        "FedLaw": {
            "config": config,
            "model": model,
            "test_data": test_data,
            "size_weights": size_weights,
            "apply_transforms": apply_transforms_test,
        },
        "PowD": {
            "candidate_client_set": config["powd_config"]["candidate_client_set"],
        },
    }

    return getattr(__import__("mak.strategies", fromlist=[STRATEGY]), STRATEGY)(
        fraction_fit=FRACTION_FIT,
        fraction_evaluate=FRACTION_EVAL,
        min_fit_clients=MIN_CLIENTS_FIT,
        min_evaluate_clients=MIN_CLIENTS_EVAL,
        min_available_clients=NUM_CLIENTS,
        evaluate_fn=get_evaluate_fn(
            centralized_testset=test_data,
            config_sim=config,
            save_model_dir=save_model_dir,
            metrics_file=out_file_path,
            device=device,
            apply_transforms_test=apply_transforms_test,
        ),
        evaluate_metrics_aggregation_fn=weighted_average,
        on_fit_config_fn=get_fit_config_fn(config_sim=config),
        initial_parameters=fl.common.ndarrays_to_parameters(
            [val.cpu().numpy() for _, val in model.state_dict().items()]
        ),
        **kwargs.get(STRATEGY, {}),
    )


def set_seed(seed: int = 13):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # Set the seed for CUDA operations (if using GPU)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    log(INFO, f"All random seeds set to {seed}")


def get_config(file_path):
    # Open the YAML file
    with open(file_path, "r") as file:
        # Parse the YAML data
        config = yaml.safe_load(file)
        return config


def get_fit_config_fn(config_sim):
    def fit_config(server_round: int):
        """Return training configuration dict for each round.

        passes the current round number to the client
        """
        config = {
            "round": server_round,
            "batch_size": config_sim["client"]["batch_size"],
            "epochs": config_sim["client"]["epochs"],
            "lr": config_sim["client"]["lr"],
            "optimizer": config_sim["common"]["optimizer"],
            "sgd_momentum": config_sim["common"]["sgd_momentum"],
            "strategy": config_sim["server"]["strategy"],
            "proximal_mu": config_sim["fedprox"]["proximal_mu"],
            "loss": config_sim["client"]["loss"],
        }
        return config

    return fit_config


def get_mode_and_shape(partition):
    data_set_keys = list(partition.features.keys())
    x_column = data_set_keys[0]
    shape = partition[x_column][0].size
    mode = partition[x_column][0].mode
    if mode == "RGB":
        channel = 3
    else:
        channel = 1
    return (channel, shape[0], shape[1])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FLNCLAB")

    parser.add_argument(
        "--config",
        type=str,
        default="./config.yaml",
        help="path to the config.yaml file.",
    )
    parser.add_argument("--strategy", type=str, help="FL Strategy/algorithm")
    parser.add_argument("--seed", type=int, help="Seed for randomness")
    parser.add_argument(
        "--noise", type=float, default=None, help="add dp noise to data or not"
    )
    parser.add_argument(
        "--dirichlet_alpha",
        type=float,
        default=None,
    )

    args = parser.parse_args()
    return args


def get_optimizer(model, client_config):
    if client_config["optimizer"] == "adam":
        return torch.optim.Adam(model.parameters(), lr=client_config["lr"])
    else:
        return torch.optim.SGD(
            model.parameters(),
            lr=client_config["lr"],
            momentum=client_config["sgd_momentum"],
        )


# for fedlaw
def get_size_weights(federated_dataset, num_clients):
    sample_size = []
    for i in range(num_clients):
        sample_size.append(len(federated_dataset.load_partition(i)))
    size_weights = [i / sum(sample_size) for i in sample_size]
    return size_weights
