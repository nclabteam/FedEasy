from functools import reduce
from logging import WARNING

from flwr.common import (
    FitRes,
    FitIns,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.common.logger import log
from flwr.common.typing import Dict, List, Optional, Tuple, Union
from flwr.server.client_proxy import ClientProxy
from flwr.server.client_manager import ClientManager
from flwr.server.strategy import FedAvg
from flwr.server.strategy.aggregate import aggregate
import numpy as np
from mak.utils.general import unmarshal_numpy, marshal_numpy

class ScaffoldStrategy(FedAvg):
    """Implement custom strategy for SCAFFOLD based on FedAvg class."""

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
        inplace=True
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
        self.global_lr = 1.0
        self.last_parameters = initial_parameters
        self.total_num_clients = self.min_available_clients
        self.global_cv = [np.zeros_like(p) for p in parameters_to_ndarrays(initial_parameters)]

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

        c_delta_results, y_delta_results = zip(*[(fit_res.metrics["c_delta"], fit_res.metrics["y_delta"]) for _, fit_res in results])
        c_delta_results = [ unmarshal_numpy(c_delta) for c_delta in c_delta_results ]
        y_delta_results = [ unmarshal_numpy(y_delta) for y_delta in y_delta_results ]

        # Aggregate weights with control
        num_clients = len(results)

        # Compute average weights of each layer
        aggregated_ndarrays = [
            global_weight + (reduce(np.add, layer_updates) / num_clients) * self.global_lr
            for *layer_updates, global_weight in zip(*y_delta_results, parameters_to_ndarrays(self.last_parameters))
        ]

        aggregated_control = [(float(num_clients)/self.total_num_clients) * reduce(np.add, c_delta) / num_clients for c_delta in zip(*c_delta_results)]

        parameters_aggregated = ndarrays_to_parameters(aggregated_ndarrays)
        self.last_parameters = parameters_aggregated
        self.global_cv = aggregated_control

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        return parameters_aggregated, metrics_aggregated
    
    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> list[tuple[ClientProxy, FitIns]]:
        """Configure the next round of training.

        Sends the proximal factor mu to the clients
        """
        # Get the standard client/config pairs from the FedAvg super-class
        client_config_pairs = super().configure_fit(
            server_round, parameters, client_manager
        )

        return [
            (
                client,
                FitIns(
                    fit_ins.parameters,
                    {**fit_ins.config, "global_control": marshal_numpy(self.global_cv)},
                ),
            )
            for client, fit_ins in client_config_pairs
        ]