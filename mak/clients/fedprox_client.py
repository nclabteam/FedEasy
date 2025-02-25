import torch

from mak.clients.base_client import BaseClient


class FedProxClient(BaseClient):
    """
    Flwr client implementation based on fedprox
    The train loop is changed based on fedprox algorithm
    """

    def __init__(
        self, client_id, model, trainset, valset, config_sim, device, save_dir
    ):
        super().__init__(
            client_id, model, trainset, valset, config_sim, device, save_dir
        )

    def __repr__(self) -> str:
        return " FedProx client"

    def train(self, net, trainloader, optim, epochs, device: str, config: dict):
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
                proximal_term = 0.0
                for local_weights, global_weights in zip(
                    net.parameters(), global_params
                ):
                    proximal_term += torch.square(
                        (local_weights - global_weights).norm(2)
                    )
                loss = (
                    criterion(net(images), labels)
                    + (config["proximal_mu"] / 2) * proximal_term
                )
                epoch_loss += loss.item()
                loss.backward()
                optim.step()
            total_loss += epoch_loss / len(trainloader)
        return total_loss  # total_loss / epochs
