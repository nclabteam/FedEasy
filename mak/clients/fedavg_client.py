from torch.utils.data import DataLoader

from mak.clients.base_client import BaseClient
from mak.utils.helper import get_optimizer


class FedAvgClient(BaseClient):
    """
    Simple flwr client implementation using basic fedavg approach
    """

    def __init__(
        self, client_id, model, trainset, valset, config_sim, device, save_dir
    ):
        super().__init__(
            client_id, model, trainset, valset, config_sim, device, save_dir
        )

    def __repr__(self) -> str:
        return " FedAvg client"

    def fit(self, parameters, config):
        self.set_parameters(parameters)

        batch, epochs, learning_rate = (
            config["batch_size"],
            config["epochs"],
            config["lr"],
        )

        trainloader = DataLoader(self.trainset, batch_size=batch, shuffle=True)
        optimizer = get_optimizer(model=self.model, client_config=config)
        self.train(
            net=self.model,
            trainloader=trainloader,
            optim=optimizer,
            epochs=epochs,
            device=self.device,
            config=config,
        )

        return self.get_parameters({}), len(trainloader.dataset), {}
