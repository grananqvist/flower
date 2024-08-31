"""Define custom Flower Ray client."""
from collections import OrderedDict
from pathlib import Path
from typing import Callable, Dict, Tuple

import flwr as fl
import numpy as np
import torch
from flwr.common.typing import NDArrays, Scalar
from torch.utils.data import DataLoader

from flair.dataset import ClientDataset
from flair.models import get_flair_model
from flair.utils import Metric, test, train


class RayClient(fl.client.NumPyClient):
    """Ray Virtual Client."""

    def __init__(
        self,
        cid: str,
        hdf5_path: Path,
        num_classes: int,
        use_fine_grained_labels: bool,
        max_num_user_images: int,
        evaluation_frequency: int,
    ):
        """Implement Ray Virtual Client.

        Parameters
        ----------
        cid : str
            Client ID, in our case a str representation of an int.
        hdf5_path : Path
            Path where partitions are saved.
        num_classes : int
            Number of classes in the classification problem.
        use_fine_grained_labels : bool
            Whether to use fine-grained or coarse-grained labels.
        max_num_user_images : int
            Limit number of images per user to this.
        evaluation_frequency : int
            Perform evaluation every this number of central iterations.
        """
        self.cid = cid
        self.hdf5_path = hdf5_path
        self.num_classes = num_classes
        self._use_fine_grained_labels = use_fine_grained_labels
        self._max_num_user_images = max_num_user_images
        self._evaluation_frequency = evaluation_frequency
        self.properties: Dict[str, Scalar] = {"tensor_type": "numpy.ndarray"}
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def get_properties(self, config: Dict[str, Scalar]) -> Dict[str, Scalar]:
        """Return properties for this client."""
        return self.properties

    def get_parameters(self, config) -> NDArrays:
        """Return weight from a given model.

        If no model is passed, then a local model is created. This can be used to
        initialize a model in the server.
        """
        net = get_flair_model(self.num_classes)
        weights = [val.cpu().numpy() for _, val in net.state_dict().items()]
        return weights

    def fit(  # type: ignore[override]
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[NDArrays, int, Metric]:
        """Fit function that performs training locally."""
        net = self.set_parameters(parameters)
        net.to(self.device)

        metrics: Metric = {}

        trainset = ClientDataset(
            hdf5_path=Path(self.hdf5_path),
            user_id=self.cid,
            partition="train",
            num_classes=self.num_classes,
            use_fine_grained_labels=self._use_fine_grained_labels,
            max_num_user_images=self._max_num_user_images,
        )
        trainloader = DataLoader(trainset, batch_size=int(config["batch_size"]))

        # Evaluate before local training.
        if config["epoch_global"] % self._evaluation_frequency == 0:
            valloader = DataLoader(trainset, batch_size=10000)
            metrics_before = test(net, valloader, device=self.device)
            metrics.update(
                {f"before training | {k}": v for k, v in metrics_before.items()}
            )

        train(
            net,
            trainloader,
            epochs=int(config["epochs"]),
            device=self.device,
            learning_rate=float(config["client_learning_rate"]),
        )

        # Evaluate after local training.
        if config["epoch_global"] % self._evaluation_frequency == 0:
            metrics_after = test(net, valloader, device=self.device)
            metrics.update(
                {f"after training | {k}": v for k, v in metrics_after.items()}
            )

        # return local model and statistics
        weights = [val.cpu().numpy() for _, val in net.state_dict().items()]
        # TODO: fix return metrics here.
        return weights, len(trainset), {}  # metrics

    def set_parameters(self, parameters: NDArrays):
        """Load weights inside the network."""
        net = get_flair_model(self.num_classes)
        weights = parameters
        params_dict = zip(net.state_dict().keys(), weights)
        state_dict = OrderedDict(
            {k: torch.from_numpy(np.copy(v)) for k, v in params_dict}
        )
        net.load_state_dict(state_dict, strict=True)
        return net


def get_ray_client_fn(
    hdf5_path: Path,
    num_classes: int,
    use_fine_grained_labels: bool,
    max_num_user_images: int,
    evaluation_frequency: int,
) -> Callable[[str], RayClient]:
    """Initialize a Ray (Virtual) Client.

    Parameters
    ----------
    hdf5_path : Path
        The FLAIR h5py dataset object.
    num_classes : int
        Number of classes in the classification problem.
    use_fine_grained_labels : bool
        Whether to use fine-grained or coarse-grained labels.
    max_num_user_images : int
        Limit number of images per user to this.
    evaluation_frequency : int
        Perform evaluation every this number of central iterations.

    Returns
    -------
    Callable[[str], RayClient]
    """

    def client_fn(cid: str) -> RayClient:
        # create a single client instance
        return RayClient(
            cid=cid,
            hdf5_path=hdf5_path,
            num_classes=num_classes,
            use_fine_grained_labels=use_fine_grained_labels,
            max_num_user_images=max_num_user_images,
            evaluation_frequency=evaluation_frequency,
        )

    return client_fn
