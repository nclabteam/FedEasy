import os
from typing import Dict

import torch
from flwr.common import Scalar
from torch.utils.data import DataLoader

from mak.clients.base_client import BaseClient
from mak.servers.scaffold_server import ScaffoldOptimizer


class ScaffoldClient(BaseClient):
    """
    Flwr client implementation based on Scaffold
    based on: https://github/adap/flower/blob/main/baselines/niid_bench/niid_bench/
    """

    def __init__(
        self, client_id, model, trainset, valset, config_sim, device, save_dir
    ):
        super().__init__(
            client_id, model, trainset, valset, config_sim, device, save_dir
        )
        # initialize client control variate with 0 and shape of the network parameters
        self.client_cv = []
        for param in self.model.parameters():
            self.client_cv.append(torch.zeros(param.shape))

    def __repr__(self) -> str:
        return " Scaffold client"

    def fit(self, parameters, config: Dict[str, Scalar]):
        batch_size, epochs, learning_rate = (
            config["batch_size"],
            config["epochs"],
            config["lr"],
        )
        momentum = 0.9
        weight_decay = 0.00001

        optim = ScaffoldOptimizer(
            self.model.parameters(), learning_rate, momentum, weight_decay
        )
        """Implement distributed fit function for a given client for SCAFFOLD."""
        # the first half are model parameters and the second are the server_cv
        server_cv = parameters[len(parameters) // 2 :]
        parameters = parameters[: len(parameters) // 2]
        self.set_parameters(parameters)
        self.client_cv = []
        for param in self.model.parameters():
            self.client_cv.append(param.clone().detach())
        # load client control variate
        if os.path.exists(f"{self.save_dir}/client_cv_{self.client_id}.pt"):
            self.client_cv = torch.load(
                f"{self.save_dir}/client_cv_{self.client_id}.pt"
            )
        # convert the server control variate to a list of tensors
        server_cv = [torch.Tensor(cv) for cv in server_cv]
        self.client_cv = [cv.detach().cpu() for cv in self.client_cv]

        trainloader = DataLoader(self.trainset, batch_size=batch_size, shuffle=True)

        self.train(
            net=self.model,
            trainloader=trainloader,
            optim=optim,
            epochs=epochs,
            device=self.device,
            config=config,
            server_cv=server_cv,
            client_cv=self.client_cv,
        )

        x = parameters
        y_i = self.get_parameters(config={})
        c_i_n = []
        server_update_x = []
        server_update_c = []
        # update client control variate c_i_1 = c_i - c + 1/eta*K (x - y_i)
        for c_i_j, c_j, x_j, y_i_j in zip(self.client_cv, server_cv, x, y_i):
            c_i_n.append(
                c_i_j
                - c_j
                + (1.0 / (learning_rate * epochs * len(trainloader))) * (x_j - y_i_j)
            )
            # y_i - x, c_i_n - c_i for the server
            server_update_x.append((y_i_j - x_j))
            server_update_c.append((c_i_n[-1] - c_i_j).cpu().numpy())
        self.client_cv = c_i_n
        torch.save(self.client_cv, f"{self.save_dir}/client_cv_{self.client_id}.pt")

        combined_updates = server_update_x + server_update_c

        return (
            combined_updates,
            len(trainloader.dataset),
            {},
        )

    def train(
        self,
        net,
        trainloader,
        optim,
        epochs,
        device: str,
        config: dict,
        server_cv,
        client_cv,
    ):
        """Train the network on the training set for fedprox."""
        criterion = self.get_loss(loss=config["loss"])
        global_params = [val.detach().clone() for val in net.parameters()]
        net.train()

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
                optim.step_custom(server_cv, client_cv)
            total_loss += epoch_loss / len(trainloader)
        return total_loss  # total_loss / epochs
