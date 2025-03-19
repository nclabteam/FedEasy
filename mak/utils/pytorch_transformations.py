from torchvision.transforms import Compose, ToTensor, Normalize, Lambda, ToPILImage, RandomCrop, RandomHorizontalFlip
import torch.nn.functional as F
from torch.autograd import Variable
from mak.utils.dataset_info import dataset_info

class TransformationPipeline:
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        self.feature_key = dataset_info[self.dataset_name]['feature_key']

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
        batch[self.feature_key] = [pytorch_transforms(img) for img in batch[self.feature_key]]
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
        batch[self.feature_key] = [pytorch_transforms(img) for img in batch[self.feature_key]]
        return batch

    def apply_transforms_default(self, batch):
        """Apply transforms to the partition from FederatedDataset."""
        pytorch_transforms = Compose(
            [
                ToTensor(),
            ]
        )
        batch[self.feature_key] = [pytorch_transforms(img) for img in batch[self.feature_key]]
        return batch

    def apply_transforms_test(self, batch):
        """Apply transforms to the partition from FederatedDataset."""
        pytorch_transforms = Compose(
            [
                ToTensor(),
            ]
        )
        batch[self.feature_key] = [pytorch_transforms(img) for img in batch[self.feature_key]]
        return batch

    def get_transformations(self):
        if self.dataset_name == "cifar10" or self.dataset_name == "cifar100":
            return self.apply_transforms_cifar10, self.apply_transforms_test
        else:
            return self.apply_transforms_default, self.apply_transforms_test