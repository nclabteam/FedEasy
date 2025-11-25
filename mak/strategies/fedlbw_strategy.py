from typing import Dict, List, Tuple, Union, Optional
from logging import WARNING
import flwr as fl
from dataclasses import dataclass, asdict
import json
from functools import reduce
import numpy as np
from math import exp
import copy

from flwr.common import (
    Scalar,
    FitRes,
    FitIns,
    Parameters,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.strategy.aggregate import aggregate
from flwr.common.logger import log
from flwr.server.client_proxy import ClientProxy
from flwr.common import NDArray, NDArrays
from mak.utils.general import set_params, test
from datasets.utils.logging import disable_progress_bar
from torch.utils.data import DataLoader

class FedLBWStrategy(fl.server.strategy.FedAvg):
    """Implement custom strategy for FedLBW based on.
    based on: https://doi.org/10.1016/j.eswa.2025.130487
    """
    def __init__(
        self,
        model,
        test_data,
        config,
        fraction_fit: float,
        fraction_evaluate: float,
        min_fit_clients: int,
        min_evaluate_clients : int,
        min_available_clients : int,
        evaluate_fn,
        evaluate_metrics_aggregation_fn,
        apply_transforms,
        device = 'cpu',
        on_fit_config_fn = None,
        **kwargs
    ) -> None:
        super().__init__(fraction_fit=fraction_fit,
                         fraction_evaluate = fraction_evaluate,
                         min_fit_clients = min_fit_clients,
                         min_evaluate_clients = min_evaluate_clients,
                         min_available_clients = min_available_clients,
                         evaluate_fn = evaluate_fn,
                         on_fit_config_fn = on_fit_config_fn)
        print("++++++++++++++++ FedLBW Strategy +++++++++++++++++++++++++++")
        self.model = model
        self.test_data = test_data
        self.fraction_fit = fraction_fit
        self.evaluate_fn = evaluate_fn
        self.on_fit_config_fn = on_fit_config_fn
        self.evaluate_metrics_aggregation_fn = evaluate_metrics_aggregation_fn
        self.apply_transforms = apply_transforms
        self.device = device
        self.config = config
        
        
    def _get_valid_set(self):
        client_dataset_splits = self.test_data.train_test_split(test_size= 0.06)
        valset = client_dataset_splits["test"]
        # # Now we apply the transform to each batch.
        valset = valset.with_transform(self.apply_transforms)
        val_loader = DataLoader(valset, batch_size=64, shuffle=True)
        return val_loader

    def __repr__(self) -> str:
        return " FedLBW Strategy"

    
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
        val_loader = self._get_valid_set()
        client_model = copy.deepcopy(self.model)
        # Convert results 
        #weighted results based on the loss
        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples, self.evaluate_client(client_model, parameters_to_ndarrays(fit_res.parameters),self.device,val_loader))
            for _, fit_res in results
        ]
        parameters_aggregated = ndarrays_to_parameters(self._aggregate(weights_results))

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        return parameters_aggregated, metrics_aggregated
    

    def evaluate_client(self, client_model, client_parameters,device, val_loader):
        set_params(client_model, client_parameters)
        client_model.to(device)
        loss, accuracy = test(client_model, val_loader, loss=self.config["client"]["loss"], device=device)
        return 1/loss

    def _aggregate(self, results: List[Tuple[NDArrays, int, float]]) -> NDArrays:
        """Compute weighted average."""
        # Calculate the total number of examples used during training
        # print(f"@@@@++++++ inside _aggregate+++++++++")
        num_examples_total = sum([num_examples for _, num_examples, _ in results])
        total_loss = sum([loss for _ , _, loss in results])

        # Create a list of weights, each multiplied by the related number of examples
        weighted_weights = [
            [layer * eval_loss for layer in weights] for weights, num_examples, eval_loss in results
        ]

        # Compute average weights of each layer
        weights_prime: NDArrays = [
            reduce(np.add, layer_updates) / total_loss
            for layer_updates in zip(*weighted_weights)
        ]
        return weights_prime

    
        