# FedEasy: Federated Learning With Ease

![FedEasy Logo](./docs/imgs/FedEasy_overview.svg)


FedEasy is an intuitive powerful yet simple to use Federated Learning framework built on top of [Flower](https://flower.ai/) framework and [PyTorch](https://pytorch.org/). Our goal is to streamline the process of setting up and running federated learning experiments, making advanced machine learning techniques accessible to researchers and developers alike.

This repository provides an easy-to-use Federated Learning framework based on Flower and PyTorch. It's designed to simplify the process of setting up and running federated learning experiments.

> **Note**: This code has been written and tested on Ubuntu and should work seamlessly on any Linux-based distribution. Windows users may need to adjust some steps.

## 📋 Prerequisites

- Git
- Python 3.x
- pip (Python package installer)
- Miniconda (recommended for environment management)

## 🚀 Getting Started

Follow these steps to set up and use this repository:

### 1. Clone the Repository

```bash
git clone https://github.com/nclabteam/FedEasy.git
```
2. After clone cd into cloned directory and open terminal.

3. Ensure pip is installed if not install using
```bash
 sudo apt install python3-pip
```

4. We will be using miniconda to create a virtual environment, download miniconda as
If You already have conda installed skip step 4 and 5.

```bash
 curl https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o Miniconda3-latest-Linux-x86_64.sh
```
5. Then install miniconda using below command
```bash
  bash Miniconda3-latest-Linux-x86_64.sh
```
6. Create a new virtual environment using conda
```bash
 conda env create -f environment.yaml
```
It will create a virtual env named `venv-fedeasy` based on `environment.yaml` file

7. For using virtual environment we need to activate the environment first.
```bash
 conda deactivate
 conda activate venv-fedeasy
```
8. We can change the confugration as per our need in config.yaml file

9. We can run scale clients to few hundred we can run flower in simulation mode on single machine like:
```bash
python main.py
```
  This script will read the confugration from `config.yaml` file and starts the simulation.

  The outputs will be saved in `out` directory.


## Description about  [`config.yaml`](/config.yaml) file
The `config.yaml` file is a configuration file for this framework that trains a Federated Learning model.

The configuration file is divided into three sections: `common`, `server`, and `client`.

### Common Section
The `common` section contains the common configurations used in this framework. 

- `data_type` : This field specifies the data distribution type used in the training process. Currently supported data distributions are [`iid`,`dirichlet_niid`] . Detailed explination can be found [here](./docs/data_distribution.md)
- `dataset` : This field specifies the dataset used in the training process. Currently supported data distributions are [`mnist`, `cifar10`, `fashion_mnist`, `sasha/dog-food`, `zh-plus/tiny-imagenet`]. Detailed explination can be found [here](./docs/datasets.md)
- `dirichlet_alpha` : This field is used when `data_type` is set to `dirichlet_niid`. It specifies the Dirichlet concentration parameter.
- `target_acc` : This field specifies the target accuracy that the model needs to achieve. It can take any value greater than `0`.
- `model` : This field specifies the model architecture used in the training process. Currently Implemented models are [ `Net`, `CifarNet`, `SimpleCNN`, `KerasExpCNN`, `MNISTCNN`, `SimpleDNN`, `FMCNNModel`,`FedAVGCNN`,`Resnet18`, `Resnet34`,`ResNet18Pretrained`, `ResNet34Pretrained`,`ResNet18Small`, `ResNet20Small`,`MobileNetV2`,`EfficientNetB0`,`LSTMModel`]. Detailed explination can be found [here](./docs/models.md)
- `optimizer` : This field specifies the optimizer used in the training process. It could be either `sgd` or `adam`.
- `seed` : This field fixes the seed for reproducibility

### Server Section
The `server` section contains the configurations for the server that coordinates the Federated Learning process.

- `max_rounds` : This field specifies the maximum number of rounds for the training process.
- `address` : This field specifies the IP address of the server.
- `fraction_fit` : This field specifies the fraction of participating clients used for training in each round.
- `min_fit_clients` : This field specifies the minimum number of participating clients required for training in each round.
- `num_clients` : Total number of clients participating in training.
- `fraction_evaluate` : This field specifies the fraction of participating clients used for evaluation in each round.
- `min_avalaible_clients` : This field specifies the minimum number of clients that should be available for the training process.
- `strategy` : This field specifies the strategy used for Federated Learning. Currently supported strategies are [`FedLaw`, `FedProx`, `FedAvgM`, `FedOpt`, `FedAdam`, `FedMedian`, `FedAvg`,] Detailed explination can be found [here](./docs/strategies.md)

### Client Section
The `client` section contains the configurations for the clients participating in the Federated Learning process.

- `epochs` : This field specifies the number of epochs for each client's training process.
- `batch_size` : This field specifies the batch size for each client's training process.
- `lr` : This field specifies the learning rate for each client's training process.
- `save_train_res` : This field specifies whether to save the training results. It could be either `true` or `false`.
If `save_train_res` is set to `true`, all the output data like accuracy, loss, time of each round would be saved in the `out` directory.
- `total_cpus` : No. of CPU cores that are assigned for all simulation
- `total_gpus` :  No. of GPU's assigned for whole simulation
- `gpu` : True or False, Use GPU for training or not. Default to False
- `num_cpus` : No. of CPU cores that are assigned for each actor Default to 1
- `num_gpus` : Fraction of GPU assigned to each actor. (num_cpus and num_gpus can only used in simulation mode if `simulation` is set to `True`) For more details on this please refer to https://flower.dev/docs/framework/how-to-run-simulations.html and https://docs.ray.io/en/latest/ray-core/scheduling/resources.html


### Code Formatting
# Installation
```
pip install isort black
```
# Usage
```
isort mak --profile=black
black mak
```

## Citation

If you use FedEasy in your research, please cite our work:

```bibtex
@software{FedEasy2025,
  author = {Kundroo, Majid and Haider, Ghani and Khoa, Nguyen and Mamond, Abdul Wahab and Kim, Taehong},
  title = {FedEasy: Federated Learning With Ease},
  year = {2025},
  url = {https://github.com/nclabteam/FedEasy},
  version = {1.0.0}
}
```