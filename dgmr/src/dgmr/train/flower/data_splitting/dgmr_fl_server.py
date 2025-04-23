import argparse
from flwr.server import ServerConfig, start_server
from flwr.server.strategy import FedAvg
from flwr.common import ndarrays_to_parameters


from dgmr.dgmr_es import DGMR


# TODO: Hand parameters below to Clients
# def parse_args():
#     parser = argparse.ArgumentParser(description='Training script')
#     parser.add_argument('--precision', type=int, default=32, help='Precision for training')
#     parser.add_argument('--gen_steps', type=int, default=2, help='Number of generation steps')
#     parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
#     parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
#     parser.add_argument('--model_name', type=str, required=True, help='Model name')
#     parser.add_argument('--num_nodes', type=int, default=1, help='Number of nodes')
#     parser.add_argument('--gpus', type=int, default=1, help='Number of GPUs')
#     return parser.parse_args()

# class SaveModelStrategy(FedAvg):
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
#         self.round = 0
#
#     def aggregate_fit(
#         self,
#         rnd: int,
#         results: List[Tuple[Any, Dict[str, Any]]],
#         failures: List[Any],
#     ) -> Optional[ndarrays_to_parameters]:
#         aggregated_parameters = super().aggregate_fit(rnd, results, failures)
#
#         if aggregated_parameters is not None:
#             self.save_global_model(aggregated_parameters, rnd)
#         else:
#             print(f"Round {rnd}: Aggregation failed.")
#
#         return aggregated_parameters
#
#     def save_global_model(self, parameters, rnd):
#         param_ndarrays = parameters_to_ndarrays(parameters)
#         state_dict = self.ndarrays_to_state_dict(param_ndarrays)
#         model = DGMR(generation_steps=2)
#         model.load_state_dict(state_dict, strict=True)
#         checkpoint_path = f"global_model_round_{rnd}.ckpt"
#         torch.save(model.state_dict(), checkpoint_path)
#         print(f"Saved global model checkpoint at round {rnd} to {checkpoint_path}")
#
#     def ndarrays_to_state_dict(self, ndarray_list):
#         model = DGMR(generation_steps=2)
#         state_keys = list(model.state_dict().keys())
#         state_dict = OrderedDict({k: torch.tensor(v) for k, v in zip(state_keys, ndarray_list)})
#         return state_dict


def get_initial_parameters():
    """get init config for global model"""
    model = DGMR(generation_steps=2)  # TODO: keep same with client
    # convert model weights to lists
    parameters = [val.cpu().numpy() for _, val in model.state_dict().items()]
    return parameters


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flower Server")
    parser.add_argument("--server_address", type=str, required=True, help="Server address")
    parser.add_argument("--num_rounds", type=int, default=3, help="Number of federated learning rounds")
    parser.add_argument(
        "--min_fit_clients", type=int, default=2, help="Minimum number of clients to participate in the training"
    )
    parser.add_argument("--min_available_clients", type=int, default=2, help="Minimum number of total clients")
    args = parser.parse_args()

    # get init model parameters
    initial_parameters = ndarrays_to_parameters(get_initial_parameters())

    # define federated learning strategy
    strategy = FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=args.min_fit_clients,
        min_available_clients=args.min_available_clients,
        initial_parameters=initial_parameters,
    )

    # start server
    start_server(
        server_address=args.server_address,
        config=ServerConfig(num_rounds=args.num_rounds),
        strategy=strategy,
        grpc_max_message_length=1024 * 1024 * 1024,
    )

"""
Sell CMD example:

1. Server:
python server.py --server_address "0.0.0.0:8080" --num_rounds 3 --min_fit_clients 2 --min_available_clients 2

2. Clinet(s):
python client.py --client_id 0 --server_address "server_ip:8080"
python client.py --client_id 1 --server_address "server_ip:8080"


python dgmr_fl_client.py --client_id 1 --server_address "127.0.0.1:8080"
"""
