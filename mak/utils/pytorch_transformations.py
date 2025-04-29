import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Lambda,
    Normalize,
    Resize,
    ToPILImage,
    ToTensor,
)

from mak.utils.dataset_info import dataset_info


class TransformationPipeline:
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        self.feature_key = dataset_info[self.dataset_name]["feature_key"]
        self.img_shape = dataset_info[self.dataset_name]["input_shape"]

    def apply_transforms_scaffold(self, batch):
        """Apply transforms to the partition from FederatedDataset.
        Transformations based on scaffold flwr baseline implementation
        """
        pytorch_transforms = Compose(
            [
                ToTensor(),
                Lambda(
                    lambda x: F.pad(
                        Variable(x.unsqueeze(0), requires_grad=False),
                        (4, 4, 4, 4),
                        mode="reflect",
                    ).data.squeeze()
                ),
                ToPILImage(),
                RandomCrop(32),
                RandomHorizontalFlip(),
                ToTensor(),
            ]
        )
        batch[self.feature_key] = [
            pytorch_transforms(img) for img in batch[self.feature_key]
        ]
        return batch

    def apply_transforms_cifar10(self, batch):
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
        batch[self.feature_key] = [
            pytorch_transforms(img) for img in batch[self.feature_key]
        ]
        return batch

    def apply_transforms_dogfood(self, batch):
        """Apply transforms to the partition from FederatedDataset sasha/dogfood."""
        pytorch_transforms = Compose(
            [
                CenterCrop(self.img_shape[1]),  # Center crop first
                Resize(self.img_shape[1]),
                ToTensor(),
            ]
        )
        batch[self.feature_key] = [
            pytorch_transforms(img) for img in batch[self.feature_key]
        ]
        return batch

    def apply_transforms_usps(self, batch):
        """Apply transforms to the partition from FederatedDataset flwrlabs/usps."""
        pytorch_transforms = Compose(
            [
                Resize(self.img_shape[1]),
                ToTensor(),
            ]
        )
        batch[self.feature_key] = [
            pytorch_transforms(img) for img in batch[self.feature_key]
        ]
        return batch

    def apply_transforms_tiny_imagenet(self, batch):
        """Apply transforms to the partition from FederatedDataset zh-plus/tiny-imagenet."""
        pytorch_transforms = Compose(
            [
                Lambda(
                    lambda img: img.convert("RGB") if img.mode != "RGB" else img
                ),  # Convert grayscale to RGB
                ToTensor(),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        batch[self.feature_key] = [
            pytorch_transforms(img) for img in batch[self.feature_key]
        ]
        return batch

    def apply_transforms_tiny_imagenet_test(self, batch):
        """Apply transforms to the partition from FederatedDataset zh-plus/tiny-imagenet."""
        pytorch_transforms = Compose(
            [
                Lambda(
                    lambda img: img.convert("RGB") if img.mode != "RGB" else img
                ),  # Convert grayscale to RGB
                ToTensor(),
            ]
        )
        batch[self.feature_key] = [
            pytorch_transforms(img) for img in batch[self.feature_key]
        ]
        return batch

    def apply_transforms_default(self, batch):
        """Apply transforms to the partition from FederatedDataset."""
        pytorch_transforms = Compose(
            [
                ToTensor(),
            ]
        )
        batch[self.feature_key] = [
            pytorch_transforms(img) for img in batch[self.feature_key]
        ]
        return batch

    def apply_transforms_test(self, batch):
        """Apply transforms to the partition from FederatedDataset."""
        pytorch_transforms = Compose(
            [
                ToTensor(),
            ]
        )
        batch[self.feature_key] = [
            pytorch_transforms(img) for img in batch[self.feature_key]
        ]
        return batch

    def get_transformations(self):
        if self.dataset_name == "cifar10" or self.dataset_name == "cifar100":
            return self.apply_transforms_cifar10, self.apply_transforms_test
        elif self.dataset_name == "sasha/dog-food":
            return self.apply_transforms_dogfood, self.apply_transforms_dogfood
        elif self.dataset_name == "flwrlabs/usps":
            return self.apply_transforms_usps, self.apply_transforms_usps
        elif self.dataset_name == "zh-plus/tiny-imagenet":
            return (
                self.apply_transforms_tiny_imagenet,
                self.apply_transforms_tiny_imagenet_test,
            )
        else:
            return self.apply_transforms_default, self.apply_transforms_test
