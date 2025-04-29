from typing import List, Tuple

import flwr as fl
from flwr.common import EvaluateIns, EvaluateRes, FitIns, Parameters
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy


class PowD(fl.server.strategy.FedAvg):
    """Power Of Choice Client Selection Algorithm.
    Towards Understanding Biased Client Selection in Federated Learning (AISTATS-2022)
    https://proceedings.mlr.press/v151/jee-cho22a.html.

    """

    def __init__(
        self,
        fraction_fit: float,
        fraction_evaluate: float,
        min_fit_clients: int,
        min_evaluate_clients: int,
        min_available_clients: int,
        evaluate_fn,
        on_fit_config_fn,
        candidate_client_set,  # d parameter for power of selection
        **kwargs,
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

        # sample size for random selection
        self.d_choice = candidate_client_set

    def __repr__(self) -> str:
        return "Pow-D"

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training.

        Apply Power-Of-Selection strategy for client selection.
        """
        config = {}
        if self.on_fit_config_fn is not None:
            # Custom fit config function provided
            config = self.on_fit_config_fn(server_round)
        fit_ins = FitIns(parameters, config)

        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )

        # Get client and client configs based on random selection
        clients = client_manager.sample(
            num_clients=self.d_choice,
            min_num_clients=min_num_clients,
        )

        # Evaluate (get losses of) d-chosen clients on global model
        eval_config = {}
        if self.on_evaluate_config_fn is not None:
            eval_config = self.on_evaluate_config_fn(server_round)
        evaluate_ins = EvaluateIns(parameters, eval_config)

        client_losses = []
        for client in clients:
            # Train clients and get loss
            client_res: EvaluateRes = client.evaluate(evaluate_ins, None)
            client_losses.append((client, client_res.loss))

        # Sort client according to higher losses
        sorted_clients = sorted(client_losses, key=lambda x: x[1], reverse=True)
        # Get sample_size (client fraction for fit) clients with highest loss
        sorted_clients = sorted_clients[:sample_size]
        # Get chosen client proxy
        chosen_clients = [x[0] for x in sorted_clients]

        # Return client/config pairs
        return [(client, fit_ins) for client in chosen_clients]
