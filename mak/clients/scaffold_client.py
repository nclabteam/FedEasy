import os
import numpy as np
from typing import Dict, OrderedDict

import torch
from flwr.common import Scalar
from torch.utils.data import DataLoader

from mak.clients.base_client import BaseClient
from mak.utils.general import get_loss, unmarshal_numpy, marshal_numpy
from mak.utils.helper import get_optimizer


class ScaffoldClient(BaseClient):
    """
    Flwr client implementation based on Scaffold
    based on: https://github/adap/flower/blob/main/baselines/niid_bench/niid_bench/
    and https:///github.com/wittenator/flower/blob/rework_fedprox_baseline/baselines/scaffold/scaffold/client_app.py
    """

    def __init__(
        self, client_id, model, trainset, valset, config_sim, device, save_dir
    ):
        super().__init__(
            client_id, model, trainset, valset, config_sim, device, save_dir
        )
        self.client_cv =  [np.zeros_like(p.cpu().detach().numpy()) for k, p in self.model.state_dict().items()]
    def __repr__(self) -> str:
        return " Scaffold client"
    

    def fit(self, parameters, config: Dict[str, Scalar]):
        batch_size, epochs, learning_rate = (
            config["batch_size"],
            config["epochs"],
            config["lr"],
        )
        self.set_parameters(parameters)
        global_cv = unmarshal_numpy(config["global_control"])
        optimizer = get_optimizer(model=self.model, client_config=config)
        trainloader = DataLoader(self.trainset, batch_size=batch_size, shuffle=True)

        loss = self.train(
            net=self.model,
            trainloader=trainloader,
            optim=optimizer,
            epochs=epochs,
            device=self.device,
            config=config,
            server_cv=global_cv,
            client_cv=self.client_cv,
        )

        # update local control
        with torch.no_grad():
            y_delta = []
            c_plus = []
            c_delta = []

            for x, y_i in zip(parameters, self.model.state_dict().values()):
                y_delta.append((y_i.cpu() - x).numpy())

            coef = 1 / (epochs * learning_rate)
            for c, c_i, y_del in zip(global_cv, self.client_cv, y_delta):
                c_plus.append(c_i - c - coef * y_del)

            for c_p, c_l in zip(c_plus, self.client_cv):
                c_delta.append(c_p - c_l)

            self.client_cv = c_plus
        return (
            self.get_parameters({}),
            len(trainloader.dataset),
            {"train_loss": loss, "c_delta": marshal_numpy(c_delta) , "y_delta": marshal_numpy(y_delta)},
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
        criterion = get_loss(loss=config["loss"])
        net.train()

        total_loss = 0
        for _ in range(epochs):
            epoch_loss = 0
            for batch in trainloader:
                keys = list(batch.keys())
                x_label, y_label = keys[0], keys[1]
                images, labels = batch[x_label].to(device), batch[y_label].to(device)
                # Skip batches that have only one sample (BatchNorm can't handle these)
                if images.size(0) == 1:
                    continue
                optim.zero_grad()
                loss = criterion(net(images), labels)
                loss.backward()
                
                for (name,param), c, c_i in zip(
                    net.state_dict().items(), server_cv, client_cv
                ):
                    if param.requires_grad:
                        # The global control does not have batchnorm dimensions at the beginning, but is zero at this point in time
                        if c.shape == c_i.shape:
                            param.grad.data += torch.tensor(c - c_i).to(device)
                        else:
                            param.grad.data += torch.tensor(-c_i).to(device)
                optim.step()
                
            total_loss += epoch_loss / len(trainloader)
        return total_loss  # total_loss / epochs
