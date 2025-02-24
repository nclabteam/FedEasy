# Config
<!-- # Description about  [`config.yaml`](/config.yaml) file -->
The `config.yaml` file is a configuration file for this framework that trains a Federated Learning model.

The configuration file is divided into three sections: `common`, `server`, and `client`.

### Common Section
The `common` section contains the common configurations used in this framework. 

- `data_type` : This field specifies the data distribution type used in the training process. Currently supported data distributions are [`iid`,`dirichlet_niid`] . Detailed explination can be found [here](./data_distribution.md)
- `dataset` : This field specifies the dataset used in the training process. Currently supported data distributions are [`mnist`, `cifar10`, `fashion_mnist`, `sasha/dog-food`, `zh-plus/tiny-imagenet`]. Detailed explination can be found [here](./datasets.md)
- `dirichlet_alpha` : This field is used when `data_type` is set to `dirichlet_niid`. It specifies the Dirichlet concentration parameter.
- `target_acc` : This field specifies the target accuracy that the model needs to achieve. It can take any value greater than `0`.
- `model` : This field specifies the model architecture used in the training process. Currently Implemented models are [ `Net`, `CifarNet`, `SimpleCNN`, `KerasExpCNN`, `MNISTCNN`, `SimpleDNN`, `FMCNNModel`,`FedAVGCNN`,`Resnet18`, `Resnet34`,`ResNet18Pretrained`, `ResNet34Pretrained`,`ResNet18Small`, `ResNet20Small`,`MobileNetV2`,`EfficientNetB0`,`LSTMModel`]. Detailed explination can be found [here](./models.md)
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
- `strategy` : This field specifies the strategy used for Federated Learning. Currently supported strategies are [`FedLaw`, `FedProx`, `FedAvgM`, `FedOpt`, `FedAdam`, `FedMedian`, `FedAvg`,`PowD`, `Scaffold`, `FedNova`,] Detailed explination can be found [here](./strategies.md)

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

### Sample Config.yaml


```
---
# config

common:
  data_type : dirichlet_niid # data_type = data distribution one among ['iid','dirichlet_niid']
  dataset : cifar10 # data_set = data set used  one among ['cifar10', 'cifar100', 'mnist', 'fashion_mnist', 'sasha/dog-food', 'zh-plus/tiny-imagenet', 'Mike0307/MNIST-M', 'flwrlabs/usps']
  dirichlet_alpha : 0.1 #dirichlet concentration parameter (only used if data_type is dirchlet-niid)
  target_acc : 0.95
  model : Net # one among [ Net, CifarNet, SimpleCNN, KerasExpCNN, MNISTCNN, SimpleDNN, FMCNNModel,FedAVGCNN,Resnet18, Resnet34,ResNet18Pretrained, ResNet34Pretrained,ResNet18Small, ResNet20Small,MobileNetV2,EfficientNetB0,LSTMModel]
  optimizer : sgd # one among [sgd,adam]
  sgd_momentum : 0.9
  seed : 8  #8,9,10
  multi_node : False
  save_log : True

server:
  num_rounds : 3
  address : 127.0.0.1
  fraction_fit : 0.1
  min_fit_clients: 2
  num_clients : 20  # total number of clients participating in training
  fraction_evaluate : 0.2
  strategy : FedAvg  #[FedLaw, FedProx, FedAvgM, FedOpt, FedAdam, FedMedian, FedAvg,PowD,Scaffold,FedNova]

client:
  epochs : 3
  batch_size : 128
  test_batch_size : 128
  lr: 0.08 #  [0.0001,0.001,0.005,0.01,0.1,0.2]
  save_train_res : True
  total_cpus : 2 # no. of CPU cores that are assigned for all simulation
  total_gpus : 1 # no. of GPU's assigned for whole simulation
  gpu : False  # True or False, Use GPU for training or not.
  num_cpus : 1  # no. of CPU cores that are assigned for each actor
  num_gpus : 0.25 #  no. of GPU that are assigned for each actor (it can be fraction value as well)

fedlaw_config:
  server_funct : 'exp' #['exp' or 'quad'] #currently exp only is implemented
  server_optimizer : 'adam'
  server_valid_ratio : 0.02 #percentage of data used for server side retraining
  server_epochs : 10

fedprox:
  proximal_mu : 0.5



```