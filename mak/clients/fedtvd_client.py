import torch
from torch.utils.data import DataLoader
from mak.clients.base_client import BaseClient
from mak.utils.helper import get_optimizer
from mak.utils.general import get_loss
from mak.utils.dataset_info import dataset_info


class FedTVDClient(BaseClient):
    """
    Flwr client implementation based on FedTVD
    based on: https://doi.org/10.1016/j.future.2025.108177
    """

    def __init__(
        self, client_id, model, trainset, valset, config_sim, device, save_dir
    ):
        super().__init__(
            client_id, model, trainset, valset, config_sim, device, save_dir
        )
        self.dataset_name = config_sim["common"]["dataset"]
        self.num_classes_dataset = dataset_info[self.dataset_name]["num_classes"]

    def __repr__(self) -> str:
        return " FedTVD Client"
    
    def fit(self, parameters, config):
        self.set_parameters(parameters)

        batch, epochs, learning_rate = (
            config["batch_size"],
            config["epochs"],
            config["lr"],
        )
        trainloader = DataLoader(self.trainset, batch_size=batch, shuffle=True)
        optimizer = get_optimizer(model=self.model, client_config=config)
        
        q = [1/self.num_classes_dataset] * self.num_classes_dataset  # uniform distribution over classes
        class_counts = [0] * self.num_classes_dataset
        for batch in trainloader:
            for label in batch[list(batch.keys())[1]]:
                class_counts[label.item()] += 1
        total_samples = 1 + sum(class_counts)
        p = [i/total_samples for i in class_counts]
        
        scale_val = 0.5 * sum(abs(p_i - q_i) for p_i, q_i in zip(p, q))
        
        self.train(
            net=self.model,
            trainloader=trainloader,
            optim=optimizer,
            epochs=epochs,
            device=self.device,
            config=config,
        )

        return self.get_parameters({}), len(trainloader.dataset), {"scale_val": scale_val}