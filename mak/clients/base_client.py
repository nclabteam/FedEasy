import os

import flwr as fl
from torch.utils.data import DataLoader

from mak.utils.general import set_params, test
from mak.utils.helper import get_optimizer


class BaseClient(fl.client.NumPyClient):
    """flwr base client implementaion"""

    def __init__(
        self,
        client_id,
        model,
        trainset,
        valset,
        config_sim,
        device,
        save_dir,
    ):
        self.client_id = client_id
        self.config_sim = config_sim
        self.trainset = trainset
        self.valset = valset
        self.model = model
        self.device = device
        self.train_batch_size = self.config_sim["client"]["batch_size"]
        self.test_batch_size = config_sim["client"]["test_batch_size"]
        self.training_mode = config_sim["client"]["training_mode"]
        self.finetune_mode = config_sim["client"]["finetune_mode"]
        self.finetune_epochs = config_sim["client"]["finetune_epochs"]
        self.model.to(self.device)
        self.save_dir = os.path.join(save_dir, "clients")

    def __repr__(self) -> str:
        return " Flwr base client"

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        set_params(self.model, parameters)

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
            training_mode=self.training_mode,
            device=self.device,
            config=config,
        )

        return self.get_parameters({}), len(trainloader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        if self.finetune_epochs >= 1:
            trainloader = DataLoader(
                self.trainset,
                batch_size=self.train_batch_size,
                shuffle=True,
            )
            optimizer = get_optimizer(self.model, config)
            self.train(
                net=self.model,
                trainloader=trainloader,
                optim=optimizer,
                epochs=self.finetune_epochs,
                training_mode=self.finetune_mode,
                device=self.device,
                config=config,
            )
        valloader = DataLoader(self.valset, batch_size=self.test_batch_size)
        loss, accuracy = self.test(self.model, valloader, device=self.device)
        return float(loss), len(valloader.dataset), {"accuracy": float(accuracy)}

    def get_loss(self, loss):
        return getattr(__import__("mak.losses", fromlist=[loss]), loss)()

    def train(
        self, net, trainloader, optim, epochs, training_mode, device: str, config: dict
    ):
        """Train the network on the training set."""
        criterion = self.get_loss(loss=config["loss"])
        self._freeze_model_if_needed(net, training_mode)
        net.train()

        for _ in range(epochs):
            for batch in trainloader:
                keys = list(batch.keys())
                x_label, y_label = keys[0], keys[1]
                images, labels = batch[x_label].to(device), batch[y_label].to(device)
                optim.zero_grad()
                loss = criterion(net(images), labels)
                loss.backward()
                optim.step()

    def test(self, net, testloader, device: str):
        return test(net=net, testloader=testloader, device=device)

    def _freeze_layers(self, net, layers_to_freeze):
        for name, param in net.named_parameters():
            if name in layers_to_freeze:
                param.requires_grad = False
            else:
                param.requires_grad = True

    def _freeze_model_if_needed(self, net, mode):
        if mode == "head":
            layers_to_freeze = list(net._modules.keys())[:-1]  # freeze body layers
        elif mode == "body":
            layers_to_freeze = [list(net._modules.keys())[-1]]  # freeze head layer
        else:
            layers_to_freeze = []  # keep full model
        self._freeze_layers(net, layers_to_freeze)
