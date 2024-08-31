"""Main script to run FLAIR baseline."""
import time
from pathlib import Path

import flwr as fl
import hydra
from flwr.common.typing import Parameters
from flwr.server import ServerConfig
from hydra.utils import instantiate, to_absolute_path
from omegaconf import DictConfig, OmegaConf

from flair import utils
from flair.client import get_ray_client_fn
from flair.dataset import get_client_ids


@hydra.main(config_path="conf", config_name="base", version_base=None)
def main(cfg: DictConfig) -> None:
    """Run the baseline.

    Parameters
    ----------
    cfg : DictConfig
        An omegaconf object that stores the hydra config.
    """
    # Print parsed config
    print(OmegaConf.to_yaml(cfg))

    start_time_total = time.perf_counter()

    hdf5_path = Path(to_absolute_path(cfg.dataset.data_path))
    num_classes = len(
        utils.get_label_mapping(str(hdf5_path), cfg.dataset.use_fine_grained_labels)
    )
    clients_ids = get_client_ids(hdf5_path, "train")
    population = len(clients_ids)

    # Get centralized evaluation function
    evaluate_fn = utils.get_flair_eval_fn(
        hdf5_path=hdf5_path,
        evaluation_frequency=cfg.evaluation_frequency,
        use_fine_grained_labels=cfg.dataset.use_fine_grained_labels,
    )

    # Define client resources and ray configs
    client_resources = {
        "num_cpus": cfg.cpus_per_client,
        "num_gpus": cfg.gpus_per_client,
    }
    ray_config = {
        "num_cpus": cfg.num_cpus,
        "num_gpus": cfg.num_gpus,
    }

    on_fit_config_fn = utils.gen_on_fit_config_fn(
        client_learning_rate=cfg.client.client_learning_rate,
        epochs_per_round=cfg.client.epochs_per_round,
        batch_size=cfg.client.batch_size,
    )

    # select strategy
    initial_parameters: Parameters = utils.get_initial_parameters(num_classes)
    strategy = instantiate(
        cfg.strategy,
        fraction_fit=float(cfg.num_clients_per_round) / population,
        fraction_evaluate=0.0,
        min_fit_clients=cfg.num_clients_per_round,
        min_evaluate_clients=0,
        min_available_clients=population,
        on_fit_config_fn=on_fit_config_fn,
        evaluate_fn=evaluate_fn,
        initial_parameters=initial_parameters,
        accept_failures=False,
        fit_metrics_aggregation_fn=utils.reduce_metrics,
    )
    strategy.initial_parameters = initial_parameters
    start_time_simulation = time.perf_counter()

    client_fn = get_ray_client_fn(
        hdf5_path=hdf5_path,
        num_classes=num_classes,
        use_fine_grained_labels=cfg.dataset.use_fine_grained_labels,
        max_num_user_images=cfg.dataset.max_num_user_images,
        evaluation_frequency=cfg.evaluation_frequency,
    )

    fl.simulation.start_simulation(
        client_fn=client_fn,
        clients_ids=clients_ids,
        num_clients=population,
        client_resources=client_resources,
        config=ServerConfig(num_rounds=cfg.num_rounds),
        strategy=strategy,
        ray_init_args=ray_config,
    )

    end_time = time.perf_counter()
    print("Total simulation time (s):", end_time - start_time_simulation)
    print("Total time (s):", end_time - start_time_total)


if __name__ == "__main__":
    main()
