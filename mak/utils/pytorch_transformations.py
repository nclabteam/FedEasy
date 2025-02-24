from torchvision.transforms import Compose, Normalize, ToTensor, Lambda, ToPILImage,RandomCrop, RandomHorizontalFlip
import torch.nn.functional as F
from torch.autograd import Variable


def apply_transforms_cifar10(batch):
    """Apply transforms to the partition from FederatedDataset."""
    pytorch_transforms = Compose(
        [
            ToTensor(),
            Normalize(
                mean=[0.49139968, 0.48215827, 0.44653124],
                std=[0.24703233, 0.24348505, 0.26158768],
            ),
        ]
    )
    batch["img"] = [pytorch_transforms(img) for img in batch["img"]]
    return batch


def apply_transforms_default(batch):
    """Apply transforms to the partition from FederatedDataset."""
    pytorch_transforms = Compose(
        [
            ToTensor(),
        ]
    )
    batch["image"] = [pytorch_transforms(img) for img in batch["image"]]
    return batch

def apply_transforms_test(batch):
    """Apply transforms to the partition from FederatedDataset."""
    pytorch_transforms = Compose(
        [
            ToTensor(),
        ]
    )
    batch["img"] = [pytorch_transforms(img) for img in batch["img"]]
    return batch


def get_transformations(dataset_name):
    if dataset_name == "cifar10" or dataset_name == "cifar100":
        return apply_transforms_cifar10
    else:
        return apply_transforms_default
