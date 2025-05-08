from functools import reduce
from logging import WARNING
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.common.logger import log
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg


class LDAWA(FedAvg):
    def __init__(
        self,
        *,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        evaluate_fn: Optional[
            Callable[
                [int, NDArrays, Dict[str, Scalar]],
                Optional[Tuple[float, Dict[str, Scalar]]],
            ]
        ] = None,
        on_fit_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        on_evaluate_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        accept_failures: bool = True,
        initial_parameters: Optional[Parameters] = None,
        fit_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        inplace: bool = True,
    ) -> None:
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
        )
        self.prev_weight = None

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        if not self.prev_weight:
            parameters_aggregated, metrics_aggregated = super().aggregate_fit(
                server_round, results, failures
            )
            self.prev_weight = parameters_aggregated
            return parameters_aggregated, metrics_aggregated
        else:
            prev_weight = parameters_to_ndarrays(self.prev_weight)
            layer_ids = np.arange(len(prev_weight))
            client_weights = [parameters_to_ndarrays(r.parameters) for _, r in results]
            num_clients = len(results)
            delta_c = {}
            for i in range(len(client_weights)):
                delta_ = []
                for x, y, idx in zip(client_weights[i], prev_weight, layer_ids):
                    if len(x.shape) and len(y.shape) > 0:
                        if np.linalg.norm(x) > 0:
                            v = (x * y).sum() / (np.linalg.norm(x) * np.linalg.norm(y))
                            delta_.append((str(idx), 1.0 * v))
                            if v > 1:
                                v = 1.0
                            client_weights[i][idx] = x * v
                    else:
                        client_weights[i][idx] = client_weights[i][idx]
                delta_c[str(i)] = delta_

            parameters_aggregated: Parameters = ndarrays_to_parameters(
                [
                    reduce(np.add, layer_updates) / num_clients
                    for layer_updates in zip(*client_weights)
                ]
            )
            self.prev_weight = parameters_aggregated
            metrics_aggregated = {}
            if self.fit_metrics_aggregation_fn:
                fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
                metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
            elif server_round == 1:  # Only log this warning once
                log(WARNING, "No fit_metrics_aggregation_fn provided")

            return parameters_aggregated, metrics_aggregated
