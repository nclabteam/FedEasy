import copy
from collections import OrderedDict
from typing import List, Tuple

import flwr as fl
import torch
from flwr.common import Metrics


# borrowed from Pytorch quickstart example
def test(net, testloader, loss : str, device: str):
    """Validate the network on the entire test set."""
    criterion = get_loss(loss=loss)
    correct, loss = 0, 0.0
    net.eval()
    with torch.no_grad():
        for data in testloader:
            keys = list(data.keys())
            x_label, y_label = keys[0], keys[1]
            images, labels = data[x_label].to(device), data[y_label].to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item() * labels.size(0)  # Scale by batch size #MAK added
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
    loss = loss / len(testloader.dataset)  # Normalize by total number of samples #MAK added
    accuracy = correct / len(testloader.dataset)
    return loss, accuracy


def set_params(model: torch.nn.ModuleList, params: List[fl.common.NDArrays]):
    """Set model weights from a list of NumPy ndarrays."""
    params_dict = zip(model.state_dict().keys(), params)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Aggregation function for (federated) evaluation metrics, i.e. those returned by
    the client's evaluate() method."""
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}

def get_loss(loss):
        return getattr(__import__("mak.losses", fromlist=[loss]), loss)()

def get_unique_classes(dataloader):
    all_labels = []
    for batch in dataloader:
        keys = list(batch.keys())
        x_label, y_label = keys[0], keys[1]
        labels = batch[y_label]
        all_labels.extend(labels.numpy())  # Assuming labels are in tensor format

    unique_classes = list(set(all_labels))
    return unique_classes


def random_pertube(model, gamma):
    new_model = copy.deepcopy(model)
    for p in new_model.parameters():
        gauss = torch.normal(mean=torch.zeros_like(p), std=1)
        if p.grad is None:
            p.grad = gauss
        else:
            p.grad.data.copy_(gauss.data)

    norm = torch.norm(
        torch.stack(
            [p.grad.norm(p=2) for p in new_model.parameters() if p.grad is not None]
        ),
        p=2,
    )

    with torch.no_grad():
        scale = gamma / (norm + 1e-12)
        scale = torch.clamp(scale, max=1.0)
        for p in new_model.parameters():
            if p.grad is not None:
                e_w = 1.0 * p.grad * scale.to(p)
                p.add_(e_w)

    return new_model
