The code of all these model classes can be found inside [`mak/models`](https://github.com/nclabteam/FedEasy/tree/main/mak/models) file.

## Models
### 1. `Resnet18` 
A simple standard resnet18 architecture.
    
### 2. `ResNet18Pretrained` 
A pretrained resnet18 architecture from torchvision.

### 3. `Net` 
A simple CNN model implemented from scratch, the architecture is shown below:
```python
    def __init__(self, num_classes: int, weights = None, *args, **kwargs):
        """
        Initialize the Net model.

        Args:
            num_classes (int): The number of classes in the classification task.
            weights (str, optional): Path to the pre-trained weights file. Defaults to None.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            None
        """
        super().__init__(num_classes, *args, **kwargs)

        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
    
        if weights is not None:
            self.load_state_dict(load(weights), strict=True)
            self.pretrained = True

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the Net model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
```

### 4. `CifarNet` 
A Simple CNN model adapted from 'PyTorch: A 60 Minute Blitz for three channel input' ( [Link](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html) ):
```python
    def __init__(self, num_classes: int, weights = None, *args, **kwargs):
        """
        Initialize the CifarNet model.

        Args:
            num_classes (int): The number of classes in the classification task.
            weights (str, optional): Path to the pre-trained weights file. Defaults to None.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            None
        """
        super().__init__(num_classes, *args, **kwargs)

        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

        if weights is not None:
            self.load_state_dict(load(weights), strict=True)
            self.pretrained = True

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the CifarNet model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
```
### 5. `FedAVGCNN` 
Architecture of CNN model used in original FedAVG paper ([Paper]( https://doi.org/10.48550/arXiv.1602.05629))
```python
    def __init__(self, num_classes: int, weights = None, *args, **kwargs) -> None:
        """
        FedAVG CNN model with customizable classifier head.

        Args:
            num_classes (int): Number of output classes.
            weights (str or None): Path to pre-trained weights (if available).
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        super().__init__(num_classes, *args, **kwargs)

        self._model = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(in_features=64 * 8 * 8, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=self.num_classes)
        )

        if weights is not None:
            self.load_state_dict(load(weights), strict=True)
            self.pretrained = True
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the FedAVGCNN model.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor.
        """
        return self._model(x)
```

