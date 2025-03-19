import copy
from logging import WARNING
from typing import Dict, List, Optional, Tuple, Union

import flwr as fl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from flwr.common import (
    FitRes,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.common.logger import log
from flwr.server.client_proxy import ClientProxy
from torch.utils.data import DataLoader

from mak.utils.general import set_params


class FedLaw(fl.server.strategy.FedAvg):
    """FedLaw Strategy.
    Revisiting Weighted Aggregation in Federated Learning with Neural Networks (ICML-2023)
    https://proceedings.mlr.press/v202/li23s.html.
    Implementation based on https://github.com/ZexiLee/ICML-2023-FedLAW and https://github.com/Gaoway/FL_MPI/blob/master/server_law.py

    """

    def __init__(
        self,
        model,
        test_data,
        fraction_fit: float,
        fraction_evaluate: float,
        min_fit_clients: int,
        min_evaluate_clients: int,
        min_available_clients: int,
        evaluate_fn,
        evaluate_metrics_aggregation_fn,
        apply_transforms_test,
        size_weights,
        config,
        device="cpu",
        on_fit_config_fn=None,
        **kwargs
    ) -> None:
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            evaluate_fn=evaluate_fn,
            on_fit_config_fn=on_fit_config_fn,
        )

        self.model = model
        self.test_data = test_data
        self.fraction_fit = fraction_fit
        self.evaluate_fn = evaluate_fn
        self.on_fit_config_fn = on_fit_config_fn
        self.evaluate_metrics_aggregation_fn = evaluate_metrics_aggregation_fn
        self.apply_transforms = apply_transforms_test
        self.device = device
        self.size_weights = size_weights
        self.config = config["fedlaw_config"]
        self.config["dataset"] = config["common"]["dataset"]
        self.config["batch_size"] = config["client"]["batch_size"]
        self.valid_set = self._get_valid_set()

    def __repr__(self) -> str:
        return "FedLaw"

    def _get_valid_set(self):
        client_dataset_splits = self.test_data.train_test_split(
            test_size=self.config["server_valid_ratio"]
        )
        valset = client_dataset_splits["test"]
        # # Now we apply the transform to each batch.
        valset = valset.with_transform(self.apply_transforms)
        val_loader = DataLoader(
            valset, batch_size=self.config["batch_size"], shuffle=True
        )
        return val_loader

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}
        select_list = self._get_selected_clients(results=results)
        agg_weights = [self.size_weights[idx] for idx in select_list]
        agg_weights = [w / sum(agg_weights) for w in agg_weights]

        c_models = []

        for _, fit_res in results:
            params_ndarray = parameters_to_ndarrays(fit_res.parameters)
            c_model = copy.deepcopy(self.model)  # Perform deepcopy here
            set_params(c_model, params_ndarray)
            c_models.append(c_model)  # Append the fully prepared model to the list

        gam, optimized_weights = self._fedlaw_optimization(
            size_weights=agg_weights,
            parameters=c_models,
            central_node=self.model,
            device=self.device,
        )

        self.model = self._fedlaw_generate_global_model(
            gam=gam,
            optmized_weights=optimized_weights,
            client_params=c_models,
            central_node=self.model,
        )

        parameters_aggregated = ndarrays_to_parameters(
            [val.cpu().numpy() for _, val in self.model.state_dict().items()]
        )

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        return parameters_aggregated, metrics_aggregated

    def _get_selected_clients(self, results):
        cids = []
        for result in results:
            cids.append(int(result[0].cid))
        return cids

    def _fedlaw_optimization(self, size_weights, parameters, central_node, device):
        """
        fedlaw optimization functions for optimize both gamma and lambdas
        """
        if self.config["dataset"] == "cifar10":
            server_lr = 0.01
        else:
            server_lr = 0.005

        cohort_size = len(parameters)
        server_epochs = self.config["server_epochs"]

        optimizees = torch.tensor(
            [torch.log(torch.tensor(j)) for j in size_weights] + [0.0],
            device=device,
            requires_grad=True,
        )

        optimizee_list = [optimizees]

        if self.config["server_optimizer"] == "adam":
            optimizer = optim.Adam(optimizee_list, lr=server_lr, betas=(0.5, 0.999))
        elif self.config["server_optimizer"] == "sgd":
            optimizer = optim.SGD(optimizee_list, lr=server_lr, momentum=0.9)

        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

        for i in range(len(optimizee_list)):
            optimizee_list[i].grad = torch.zeros_like(optimizee_list[i])

        softmax = nn.Softmax(dim=0)

        central_node.train()
        for epoch in range(server_epochs):
            for itr, batch in enumerate(self.valid_set):
                for i in range(cohort_size):
                    c_model = parameters[i]
                    weight = softmax(optimizees[:-1])[i]
                    scaling_factor = torch.exp(optimizees[-1])
                    updated_params = [
                        scaling_factor * weight * new_param
                        for new_param in c_model.parameters()
                    ]

                    if i == 0:
                        model_param = updated_params
                    else:
                        model_param = [
                            mp + up for mp, up in zip(model_param, updated_params)
                        ]

                with torch.no_grad():
                    for param, new_param in zip(central_node.parameters(), model_param):
                        param.data = new_param.detach().clone()

                keys = list(batch.keys())
                x_label, y_label = keys[0], keys[1]
                data, target = batch[x_label].to(device), batch[y_label].to(device)
                optimizer.zero_grad()
                output = central_node(data)
                loss = F.cross_entropy(output, target)
                loss.backward()
                optimizer.step()
            scheduler.step()
            optmized_weights = [
                j for j in softmax(optimizees[:-1]).detach().cpu().numpy()
            ]
            learned_gamma = torch.exp(optimizees[-1])

        return learned_gamma, optmized_weights

    def _fedlaw_generate_global_model(
        self, gam, optmized_weights, client_params, central_node
    ):
        # Initialize the global model parameters dictionary
        fedlaw_param = {}

        # Convert gam to a scalar value
        gam_scalar = gam.item()

        # Iterate through client parameters to aggregate them
        for i in range(len(client_params)):
            client_param = client_params[i]
            # Convert client_param to a dictionary if it's not already
            if isinstance(client_param, dict):
                client_param_dict = client_param
            else:
                client_param_dict = client_param.state_dict()

            # Calculate the weighted parameters for the current client
            weight_scale = gam_scalar * optmized_weights[i]

            # Aggregate the parameters from the current client
            for k, v in client_param_dict.items():
                if k in fedlaw_param:
                    fedlaw_param[k] += weight_scale * v
                else:
                    fedlaw_param[k] = weight_scale * v

        # Load the aggregated parameters into the central node
        central_node.load_state_dict(fedlaw_param)

        return central_node
