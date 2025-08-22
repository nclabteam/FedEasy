from mak.clients.fedavg_client import FedAvgClient


class FedBABUClient(FedAvgClient):
    def __init__(
        self, client_id, model, trainset, valset, config_sim, device, save_dir
    ):
        super().__init__(
            client_id, model, trainset, valset, config_sim, device, save_dir
        )
        self.training_mode = "body"
