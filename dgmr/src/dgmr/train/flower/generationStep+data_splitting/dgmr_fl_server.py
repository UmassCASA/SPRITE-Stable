# server.py

import argparse
import os
from flwr.server import ServerConfig, start_server
from flwr.server.strategy import FedAvg
from flwr.server.strategy.aggregate import aggregate
from flwr.common import (
    FitIns,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from typing import Dict
from flwr.server.server import Server
from flwr.server.client_manager import SimpleClientManager
import threading
import torch
from collections import OrderedDict

from dgmr.dgmr_es_fl_gradient_checkpoint import DGMR
from train.flower.dataloader.dgmr_fl_dataloader import DGMRDataModule

import wandb
from sprite_core.config import Config

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer


def start_server_with_stop(strategy, server_address, config):
    """Start the Flower server in a separate thread and return the server object and thread."""
    # Create the client manager
    client_manager = SimpleClientManager()

    # Create the server instance
    server = Server(strategy=strategy, client_manager=client_manager)
    # Pass the server reference to the strategy
    strategy.set_server(server)

    # Start the server thread
    server_thread = threading.Thread(
        target=start_server,
        kwargs={
            "server_address": server_address,
            "config": config,
            "grpc_max_message_length": 1024 * 1024 * 1024,
            "server": server,  # Pass the server object
            "strategy": strategy,  # Pass the strategy object
        },
    )
    server_thread.start()

    return server, server_thread


def convert_metrics(metrics: dict) -> dict:
    """Convert Torch tensors in metrics to Python scalars for logging."""
    processed_metrics = {}
    for key, value in metrics.items():
        if isinstance(value, torch.Tensor):
            processed_metrics[key] = value.item()
        else:
            processed_metrics[key] = value
    return processed_metrics


class CustomStrategy(FedAvg):
    def __init__(self, min_fit_clients, min_available_clients, num_client=1, *args, **kwargs):
        super().__init__(*args, **kwargs, min_fit_clients=min_fit_clients, min_available_clients=min_available_clients)

        # Initialize server state
        self.parameters = self.get_initial_parameters()

        self.client_status = {}  # Tracks client status: "busy" or "idle"
        self.data_shard_status = {}  # Tracks data shard status
        self.max_iterations = 500  # Target iterations per data shard
        self.stop_server = False  # Flag to stop the server
        self.server_ref = None  # Reference to the server object
        self.total_shards = 4  # Total number of data shards, should match client data loading
        self.num_client = num_client

        # Initialize data shard status
        self.initialize_data_shard_status()

        # Track the best three evaluation metrics
        self.best_val_crps = float("inf")
        self.best_val_g_loss = float("inf")
        self.best_val_eval_score = float("-inf")

        self.parameter_updated = False

        # Initialize wandb
        os.makedirs(Config.WANDB_DIR / "dgmr_fl", exist_ok=True)
        wandb.init(project="DGMR_FL_dataSplit", name="server_metrics_aggregation", dir=Config.WANDB_DIR / "dgmr_fl")

    def set_server(self, server):
        """Set the server object reference."""
        self.server_ref = server

    def get_initial_parameters(self) -> Parameters:
        """Obtain initial global model parameters."""
        model = DGMR(generation_steps=6)
        parameters = [val.cpu().numpy() for _, val in model.state_dict().items()]
        return ndarrays_to_parameters(parameters)

    def initialize_parameters(self, client_manager):
        """Override if needed; return initial parameters for clients."""
        return self.get_initial_parameters()

    def should_stop(self):
        """Check if all data shards are completed."""
        completed_shards = [
            shard_info for shard_info in self.data_shard_status.values() if shard_info["status"] == "completed"
        ]
        if len(completed_shards) == len(self.data_shard_status):
            return True  # All shards completed
        return False

    def parameters_to_state_dict(self, parameters):
        """Convert Parameters object to a state_dict for the model."""
        params_ndarrays = parameters_to_ndarrays(parameters)
        model = DGMR(generation_steps=6)
        state_dict_keys = model.state_dict().keys()
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in zip(state_dict_keys, params_ndarrays)})
        return state_dict

    def save_global_model(self, model_save_path: str = "FL_global_model_t1.ckpt"):
        """Save the current global model parameters."""
        base_dir = "/work/pi_mzink_umass_edu/SPRITE/skillful_nowcasting/output/models/dataSplitting/"
        model_state_dict = self.parameters_to_state_dict(self.parameters)

        # Create a model instance
        model = DGMR(generation_steps=6)
        model.load_state_dict(model_state_dict)

        # Save model
        torch.save(model.state_dict(), base_dir + model_save_path)
        print(f"Global model saved to {base_dir + model_save_path}")

    def initialize_data_shard_status(self):
        """Initialize data shard status."""
        self.data_shard_status = {}
        for shard_id in range(self.total_shards):
            self.data_shard_status[shard_id] = {
                "status": "idle",  # shard status: "idle", "busy", or "completed"
                "iterations": 0,  # number of completed iterations
                "assigned_to": [],  # list of client IDs
                "results_received": [],
                "model_version": 0,
            }

    def configure_fit(self, server_round, parameters, client_manager):
        """Configure training tasks for clients."""
        # Get all available clients
        clients = client_manager.sample(num_clients=client_manager.num_available())

        fit_ins_list = []

        for client_proxy in clients:
            client_id = client_proxy.cid
            if client_id not in self.client_status:
                self.client_status[client_id] = "idle"
            if self.client_status.get(client_id, "idle") == "idle":
                # Assign task to idle clients
                data_shard = self.assign_data_shard(client_id)
                if data_shard is not None:
                    config = {
                        "task": "train",
                        "data_shard": data_shard,
                        "total_shards": self.total_shards,
                        "num_client": self.num_client,
                        "model_version": self.data_shard_status[data_shard]["model_version"],
                    }
                    fit_ins = FitIns(self.parameters, config)
                    fit_ins_list.append((client_proxy, fit_ins))
                    self.client_status[client_id] = "busy"
                    if len(self.data_shard_status[data_shard]["assigned_to"]) >= self.num_client:
                        self.data_shard_status[data_shard]["status"] = "busy"
                    print(f"Client status: {self.client_status}")
                    print(f"Data shard status:{self.data_shard_status}")
                else:
                    # No available shard, client remains idle
                    pass
            else:
                # Client is busy, skip
                pass
        return fit_ins_list

    def aggregate_fit(self, server_round, results, failures):
        """Aggregate loss from clients, compute average, and instruct a client to backprop if needed."""
        # Handle failed tasks
        for failed_case in failures:
            if isinstance(failed_case, BaseException):
                print(f"Exception encountered: {failed_case}")
                # Here you can handle the exception

        weights_results = []
        metrics_sum = {
            "train_eval_score": 0.0,
            "train_d_loss": 0.0,
            "train_g_loss": 0.0,
            "train_grid_loss": 0.0,
            "train_csi": 0.0,
            "train_psd": 0.0,
            "train_crps": 0.0,
        }
        num_clients = len(results)

        # Collect results from clients
        for client_res in results:
            client_id = client_res[0].cid
            res = client_res[1]

            task = res.metrics.get("task", "update")
            if task == "update":
                weights_results = [(parameters_to_ndarrays(res.parameters), res.num_examples)]

                data_shard = res.metrics["data_shard"]
                shard_info = self.data_shard_status[data_shard]
                # Update shard status
                shard_info["iterations"] += 1 / self.num_client
                shard_info["assigned_to"].remove(client_id)
                if len(shard_info["assigned_to"]) < self.num_client:
                    shard_info["status"] = "idle"
                shard_info["model_version"] += round(1 / self.total_shards, 2)
                if shard_info["iterations"] >= self.max_iterations:
                    shard_info["status"] = "completed"
                # Update client status to idle
                self.client_status[client_id] = "idle"

                # Accumulate client metrics
                for key in metrics_sum.keys():
                    metrics_sum[key] += res.metrics.get(key, 0.0)
            else:
                print("Unknown task")

        aggregated_ndarrays = aggregate(weights_results)
        parameters_aggregated = ndarrays_to_parameters(aggregated_ndarrays)
        self.parameters = parameters_aggregated
        self.parameter_updated = True
        print(f"Client status: {self.client_status}")
        print(f"Data shard status:{self.data_shard_status}")
        self.custom_evaluate(server_round)

        # Compute average metrics
        if num_clients == 0:
            num_clients = 1
        training_metrics_avg = {key: value / num_clients for key, value in metrics_sum.items()}

        print(f"Training metrics average: \n{training_metrics_avg}")
        wandb.log(training_metrics_avg, step=server_round)

        # Check if we should stop
        if self.should_stop():
            print("All data shards have completed training. Stopping training.")

            # Save global model parameters
            self.save_global_model()

            # Signal the server to stop gracefully
            self.stop_server = True

            # Call server stop
            if self.server_ref is not None:
                self.server_ref.disconnect_all_clients(5000)
                print("Server has received stop instruction.")
            else:
                print("Cannot stop server because server reference does not exist.")

        return self.parameters, {}

    def custom_evaluate(self, server_round):
        """Custom evaluation using the aggregated parameters."""
        if len(self.parameters.tensors) == 0:
            return [], {}

        val_metrics = {
            "val_eval_score": 0.0,
            "val_d_loss": 0.0,
            "val_g_loss": 0.0,
            "val_grid_loss": 0.0,
            "val_csi": 0.0,
            "val_psd": 0.0,
            "val_crps": 0.0,
        }

        batch_size = 16
        datamodule = DGMRDataModule(batch_size=batch_size)
        datamodule.setup()

        trainer = Trainer(
            logger=WandbLogger(),
            accelerator="gpu",
            devices="auto",
        )

        model_state_dict = self.parameters_to_state_dict(self.parameters)

        # Create model instance
        model = DGMR(generation_steps=6)
        model.load_state_dict(model_state_dict)
        val_metrics = trainer.validate(model, datamodule, verbose=True)

        print(f"Validation metrics average: \n{val_metrics}")
        self.update_best_metrics(val_metrics[0], server_round)

        return val_metrics[0]["val_g_loss"], val_metrics

    def update_best_metrics(self, validation_metrics_avg: Dict[str, Scalar], server_round: int):
        """Update the best metrics and save the model if a new best is achieved."""
        model_saved = False

        # Check if val_crps is the lowest
        if validation_metrics_avg["val_crps"] < self.best_val_crps:
            self.best_val_crps = validation_metrics_avg["val_crps"]
            self.save_global_model("FL_global_model_best_val_crps_round.ckpt")
            model_saved = True
            print(f"New lowest val_crps: {self.best_val_crps}, model saved.")

        # Check if val_g_loss is the lowest
        if validation_metrics_avg["val_g_loss"] < self.best_val_g_loss:
            self.best_val_g_loss = validation_metrics_avg["val_g_loss"]
            self.save_global_model("FL_global_model_best_val_g_loss.ckpt")
            model_saved = True
            print(f"New lowest val_g_loss: {self.best_val_g_loss}, model saved.")

        # Check if val_eval_score is the highest
        if validation_metrics_avg["val_eval_score"] > self.best_val_eval_score:
            self.best_val_eval_score = validation_metrics_avg["val_eval_score"]
            self.save_global_model("FL_global_model_best_val_eval_score.ckpt")
            model_saved = True
            print(f"New highest val_eval_score: {self.best_val_eval_score}, model saved.")

        if not model_saved:
            print("No new best model in this round.")

    def assign_data_shard(self, client_id):
        """Assign a data shard to a client, preferring shards with fewer iterations."""
        available_shards = [
            (shard_id, shard_info)
            for shard_id, shard_info in self.data_shard_status.items()
            if shard_info["status"] == "idle" and shard_info["iterations"] < self.max_iterations
        ]

        if not available_shards:
            return None

        # Sort by ascending iteration count
        available_shards.sort(key=lambda x: x[1]["iterations"])

        # Assign shards that need more clients
        for shard_id, shard_info in available_shards:
            if len(shard_info["assigned_to"]) < self.num_client:
                shard_info["assigned_to"].append(client_id)
                print(f"Assigned data shard {shard_id} to client {client_id}")
                return shard_id

        return None

    def __del__(self):
        """Destructor to clean up the server if needed."""
        if self.server_ref is not None:
            self.server_ref.disconnect_all_clients(5000)
            print("Server has received stop instruction.")
        else:
            print("Cannot stop server because server reference does not exist.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flower Server")
    parser.add_argument("--server_address", type=str, required=True, help="Server address")
    parser.add_argument("--num_rounds", type=int, default=3, help="Number of federated learning rounds")
    parser.add_argument(
        "--min_fit_clients", type=int, default=2, help="Minimum number of clients to participate in the training"
    )
    parser.add_argument("--min_available_clients", type=int, default=2, help="Minimum number of total clients")
    parser.add_argument("--client_number", type=int, required=True, help="Client total number, including self client")
    args = parser.parse_args()

    # Define the custom strategy
    strategy = CustomStrategy(
        min_fit_clients=args.min_fit_clients,
        min_available_clients=args.min_available_clients,
        num_client=args.client_number,
    )

    # Start the server and pass the server reference
    server_address = args.server_address
    config = ServerConfig(num_rounds=args.num_rounds)
    server, server_thread = start_server_with_stop(strategy, server_address, config)

    # Wait for the server thread to finish
    server_thread.join()
    print("Server stopped.")
