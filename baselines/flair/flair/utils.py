"""Utilities."""
from collections import OrderedDict
from functools import reduce
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import h5py
import numpy as np
import torch
from flwr.common.parameter import ndarrays_to_parameters
from flwr.common.typing import NDArrays, Parameters, Scalar
from torch.nn import Module
from torch.utils.data import DataLoader

from flair.dataset import get_label_mapping, get_multi_hot_targets
from flair.metrics import Metric
from flair.models import get_flair_model


def train(
    net: Module,
    trainloader: DataLoader,
    epochs: int,
    device: torch.device,
    learning_rate: float = 0.01,
) -> None:
    """Train the network on the training set."""
    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)
    net.train()
    for _ in range(epochs):
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = net.loss(images, labels)
            loss.backward()
            optimizer.step()


def test(net: Module, testloader: DataLoader, device: torch.device) -> Metric:
    """Validate the network on the entire test set."""
    metrics = None
    net.eval()
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            new_metrics = net.metrics(images, labels)
            metrics = add_metrics(metrics, new_metrics)
    return metrics or {}


def gen_on_fit_config_fn(
    epochs_per_round: int, batch_size: int, client_learning_rate: float
) -> Callable[[int], Dict[str, Scalar]]:
    """Generate ` On_fit_config`.

    Args:
        epochs_per_round (int):  number of local epochs.
        batch_size (int): Batch size
        client_learning_rate (float): Learning rate of clinet

    Returns
    -------
    Callable[[int], Dict[str, Scalar]]
        Function to be called at the beginnig of each rounds.
    """

    def on_fit_config(server_round: int) -> Dict[str, Scalar]:
        """Return a configuration with specific client learning rate."""
        local_config: Dict[str, Scalar] = {
            "epoch_global": server_round,
            "epochs": epochs_per_round,
            "batch_size": batch_size,
            "client_learning_rate": client_learning_rate,
        }
        return local_config

    return on_fit_config


def get_flair_eval_fn(
    hdf5_path: Path, evaluation_frequency: int, use_fine_grained_labels: bool, pretrained: bool
) -> Callable[
    [int, NDArrays, Dict[str, Scalar]], Optional[Tuple[float, Dict[str, Scalar]]]
]:
    """Return an evaluation function for centralized evaluation."""
    num_classes = len(get_label_mapping(str(hdf5_path), use_fine_grained_labels))
    inputs_all_list, targets_all_list = [], []
    with h5py.File(hdf5_path, "r") as h5:
        for user_id in h5["/val"]:
            inputs = np.array(h5[f"/val/{user_id}/images"])
            targets_shape = (len(inputs), num_classes)
            targets = get_multi_hot_targets(
                targets_shape, h5, "val", user_id, use_fine_grained_labels
            )
            inputs_all_list.append(inputs)
            targets_all_list.append(targets)

    inputs_all = torch.as_tensor(np.vstack(inputs_all_list))
    targets_all = torch.as_tensor(np.vstack(targets_all_list))
    testset = torch.utils.data.TensorDataset(inputs_all, targets_all)

    def evaluate(
        server_round: int, parameters_ndarrays: NDArrays, _config: Dict[str, Scalar]
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        if server_round % evaluation_frequency != 0:
            return None
        # determine device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        net = get_flair_model(num_classes, pretrained)
        state_dict = OrderedDict(
            {
                k: torch.tensor(np.atleast_1d(v))
                for k, v in zip(net.state_dict().keys(), parameters_ndarrays)
            }
        )
        net.load_state_dict(state_dict, strict=True)
        net.to(device)

        # Use the entire CIFAR-10 test set for evaluation."
        testloader = torch.utils.data.DataLoader(testset, batch_size=50)
        metrics = test(net, testloader, device=device)
        processed_metrics = reduce_metrics(
            [(0, {f"Central val | {k}": v for k, v in metrics.items()})]
        )
        return processed_metrics["Central val | micro loss"], processed_metrics

    return evaluate


def get_initial_parameters(num_classes, pretrained) -> Parameters:
    """Return initial parameters from a model.

    Args:
        num_classes (int, optional): Defines if using CIFAR10 or 100. Defaults to 10.
        pretrained (bool): Whether to use pre-trained model.

    Returns
    -------
        Parameters: Parameters to be sent back to the server.
    """
    model = get_flair_model(num_classes, pretrained)
    weights = [val.cpu().numpy() for _, val in model.state_dict().items()]
    parameters = ndarrays_to_parameters(weights)

    return parameters


def add_metrics(m1: Optional[Metric], m2: Metric):
    """Aggregate two metrics."""
    if m1 is None:
        return m2
    assert set(m1.keys()) == set(m2.keys())
    summed_metrics = {}
    for k in m1.keys():
        if isinstance(m1[k], tuple):
            assert isinstance(m2[k], tuple)
            summed_metrics[k] = (
                m1[k][0] + m2[k][0],
                m1[k][1] + m2[k][1],
            )
        else:
            summed_metrics[k] = m1[k] + m2[k]

    return summed_metrics


def _postprocess_metric(name, v):
    if any(name.endswith(pattern) for pattern in ("macro AP", "micro AP")):
        return v.overall_value
    else:
        print("dividing metric", name, v)
        try:
            return np.mean(np.nan_to_num(v[0] / v[1]))
        except ZeroDivisionError:
            print('div by zero!')
            return 0



def reduce_metrics(metrics: List[Tuple[int, Metric]]) -> Dict:
    """Aggregate a list of metrics."""
    summed_metrics = reduce(add_metrics, [m[1] for m in metrics], None)
    # np.mean will average across classes if macro weighted.
    return {k: _postprocess_metric(k, v) for k, v in summed_metrics.items()}
