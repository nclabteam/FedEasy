import torch
from torch.utils.data import DataLoader

from mak.clients.base_client import BaseClient
from mak.utils.helper import get_optimizer


class FedNovaClient(BaseClient):
    """
    Flwr client implementation based on FedNova
    based on: https://github/adap/flower/blob/main/baselines/niid_bench/niid_bench/
    """

    def __init__(
        self, client_id, model, trainset, valset, config_sim, device, save_dir
    ):
        super().__init__(
            client_id, model, trainset, valset, config_sim, device, save_dir
        )
        # initialize client control variate with 0 and shape of the network parameters
        self.momentum = 0.9
        self.weight_decay = 0.00001

    def __repr__(self) -> str:
        return " FedNova client"

    def fit(self, parameters, config):
        self.set_parameters(parameters)

        batch, epochs, learning_rate = (
            config["batch_size"],
            config["epochs"],
            config["lr"],
        )

        trainloader = DataLoader(self.trainset, batch_size=batch, shuffle=True)
        optimizer = get_optimizer(model=self.model, client_config=config)

        a_i, g_i = self.train(
            net=self.model,
            trainloader=trainloader,
            optim=optimizer,
            epochs=epochs,
            device=self.device,
            config=config,
        )
        g_i_np = [param.cpu().numpy() for param in g_i]
        return g_i_np, len(trainloader.dataset), {"a_i": a_i}
        # return self.get_parameters({}), len(trainloader.dataset), {}

    def train(self, net, trainloader, optim, epochs, device: str, config: dict):
        """Train the network on the training set for fedprox."""
        criterion = self.get_loss(loss=config["loss"])

        global_params = [val.detach().clone() for val in net.parameters()]
        net.train()
        local_steps = 0

        total_loss = 0
        for _ in range(epochs):
            epoch_loss = 0
            for batch in trainloader:
                keys = list(batch.keys())
                x_label, y_label = keys[0], keys[1]
                images, labels = batch[x_label].to(device), batch[y_label].to(device)
                optim.zero_grad()
                loss = criterion(net(images), labels)
                epoch_loss += loss.item()
                loss.backward()
                optim.step()
                local_steps += 1
            total_loss += epoch_loss / len(trainloader)
        # compute ||a_i||_1
        a_i = (
            local_steps
            - (self.momentum * (1 - self.momentum**local_steps) / (1 - self.momentum))
        ) / (1 - self.momentum)
        # compute g_i
        g_i = [
            torch.div(prev_param - param.detach(), a_i)
            for prev_param, param in zip(global_params, net.parameters())
        ]

        return a_i, g_i
        # return total_loss  # total_loss / epochs
