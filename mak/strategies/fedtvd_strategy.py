from functools import reduce
from logging import WARNING

import numpy as np
from flwr.common import (
    FitRes,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.common.logger import log
from flwr.common.typing import Dict, List, Optional, Tuple, Union
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg
from flwr.server.strategy.aggregate import aggregate


class FedTVDStrategy(FedAvg):
    """Implement custom strategy for FedTVD based on.
    based on: https://doi.org/10.1016/j.future.2025.108177
    """

    def __init__(
        self,
        *,
        fraction_fit=1,
        fraction_evaluate=1,
        min_fit_clients=2,
        min_evaluate_clients=2,
        min_available_clients=2,
        evaluate_fn=None,
        on_fit_config_fn=None,
        on_evaluate_config_fn=None,
        accept_failures=True,
        initial_parameters=None,
        fit_metrics_aggregation_fn=None,
        evaluate_metrics_aggregation_fn=None,
        inplace=True,
        balance_lambda=0.5,
    ):
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            evaluate_fn=evaluate_fn,
            on_fit_config_fn=on_fit_config_fn,
            on_evaluate_config_fn=on_evaluate_config_fn,
            accept_failures=accept_failures,
            initial_parameters=initial_parameters,
            fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
            evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
            inplace=inplace,
        )
        self.balance_lambda = balance_lambda

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
        
        aggregated_ndarrays = self._aggregate(results)
        parameters_aggregated = ndarrays_to_parameters(aggregated_ndarrays)

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        return parameters_aggregated, metrics_aggregated


    def _aggregate(self, results: List[Tuple[NDArrays, int, float]]) -> NDArrays:
        """Compute in-place weighted average."""
        a_clients = softmax([1/fit_res.metrics['scale_val'] for (_, fit_res) in results])
        for i in range(len(results)): 
            fit_res = results[i][1]
            fit_res.metrics['scale_val'] = a_clients[i]
        
        # Count total examples
        num_examples_total = sum(fit_res.num_examples for (_, fit_res) in results)
        scaling_factors = [
            self.balance_lambda * fit_res.metrics['scale_val'] + (1- self.balance_lambda) * fit_res.num_examples / num_examples_total for _, fit_res in results
        ] #balance_lambda an and 1-balance_lambda dynamic scaling factors
        
        # Let's do in-place aggregation
        # Get first result, then add up each other
        params = [
            scaling_factors[0] * x for x in parameters_to_ndarrays(results[0][1].parameters)
        ]
        for i, (_, fit_res) in enumerate(results[1:]):
            res = (
                scaling_factors[i + 1] * x
                for x in parameters_to_ndarrays(fit_res.parameters)
            )
            params = [reduce(np.add, layer_updates) for layer_updates in zip(params, res)]

        return params
    
    
def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def softmax_with_margin(x, alpha=1.0):
    """Compute softmax values for each set of scores in x with a margin to expand the difference."""
    # Increase the largest value by alpha to add a margin
    margin_x = x - np.max(x)  # First, normalize by subtracting the maximum value
    margin_x[np.argmax(x)] += alpha  # Add alpha to the largest value
    e_x = np.exp(margin_x)
    return e_x / e_x.sum(axis=0)