import flwr as fl
from flwr_datasets import FederatedDataset

from mak.clients.fedavg_client import FedAvgClient
from mak.clients.fedbabu_client import FedBABUClient
from mak.clients.fednova_client import FedNovaClient
from mak.clients.fedprox_client import FedProxClient
from mak.clients.scaffold_client import ScaffoldClient


def get_client_fn(
    config_sim: dict,
    dataset: FederatedDataset,
    model,
    device,
    apply_transforms,
    save_dir,
):
    strategy = config_sim["server"]["strategy"].lower()
    client_class = get_client_class(strategy)

    def client_fn(cid: str) -> fl.client.Client:
        client_dataset = dataset.load_partition(int(cid), "train")
        client_dataset_splits = client_dataset.train_test_split(
            test_size=0.2, seed=config_sim["common"]["seed"]
        )
        trainset = client_dataset_splits["train"].with_transform(apply_transforms)
        valset = client_dataset_splits["test"].with_transform(apply_transforms)
        return client_class(
            client_id=int(cid),
            model=model,
            trainset=trainset,
            valset=valset,
            config_sim=config_sim,
            device=device,
            save_dir=save_dir,
        ).to_client()

    return client_fn


def get_client_class(strategy: str):
    if strategy == "fedprox":
        return FedProxClient
    elif strategy == "scaffold":
        return ScaffoldClient
    elif strategy == "fednova":
        return FedNovaClient
    elif strategy == "fedbabu":
        return FedBABUClient
    else:
        return FedAvgClient
