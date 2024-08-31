"""Define dataset."""
import json
from pathlib import Path
from typing import Dict, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


def get_client_ids(
    hdf5_path: Path,
    partition: str,
):
    """Get the list of users from the dataset file."""
    with h5py.File(hdf5_path, "r") as h5:
        return list(h5[f"/{partition}"].keys())


def get_label_mapping(hdf5_path: str, use_fine_grained_labels: bool) -> Dict[str, int]:
    """Get the mapping of labels to indices.

    :param hdf5_path:
        The FLAIR h5py dataset object.
    :param use_fine_grained_labels:
        Whether to use fine-grained label taxonomy.
    :return:
        A dictionary with label as key and index as value
    """
    with h5py.File(hdf5_path, "r") as h5:
        if use_fine_grained_labels:
            return json.loads(h5["/metadata/fine_grained_label_mapping"][()])
        return json.loads(h5["/metadata/label_mapping"][()])


def get_multi_hot_target_one_sample(
    index: int,
    num_classes: int,
    h5: h5py.File,
    partition: str,
    user_id: str,
    use_fine_grained_labels: bool,
) -> np.ndarray:
    """Read user from `h5` and get multi-hot label for datapoint number `index`."""
    prefix = "fine_grained_labels" if use_fine_grained_labels else "labels"
    mask = np.array(h5[f"/{partition}/{user_id}/{prefix}_row"]) == index
    col_indices = np.array(h5[f"/{partition}/{user_id}/{prefix}_col"])[mask]
    vec = np.zeros(num_classes, dtype=np.float32)
    vec[col_indices] = 1
    return vec


def get_multi_hot_targets(
    shape: Tuple[int, int],
    h5: h5py.File,
    partition: str,
    user_id: str,
    use_fine_grained_labels: bool,
) -> np.ndarray:
    """Read user from `h5` and get multi-hot labels for all datapoints."""
    prefix = "fine_grained_labels" if use_fine_grained_labels else "labels"
    row_indices = np.array(h5[f"/{partition}/{user_id}/{prefix}_row"])
    col_indices = np.array(h5[f"/{partition}/{user_id}/{prefix}_col"])
    vec = np.zeros(shape, dtype=np.float32)
    vec[row_indices, col_indices] = 1
    return vec


class ClientDataset(Dataset):
    """Client Dataset."""

    def __init__(
        self,
        hdf5_path: Path,
        user_id: str,
        partition: str,
        num_classes: int,
        use_fine_grained_labels: bool,
        max_num_user_images: int,
    ):
        """Implement local dataset for FLAIR.

        Parameters
        ----------
        hdf5_path : Path
            The FLAIR h5py dataset object.
        user_id : str
            The ID of the user, must be present in dataset file.
        partition : str
            The partition `user_id` belongs to: "train", "val" or "test".
        num_classes : int
            Number of classes in the classification problem.
        use_fine_grained_labels : bool
            Whether to use fine-grained or coarse-grained labels.
        max_num_user_images : int
            Limit number of images per user to this.
        """
        super().__init__()
        self._hdf5_path = hdf5_path
        self._user_id = user_id
        self._partition = partition
        self._num_classes = num_classes
        self._use_fine_grained_labels = use_fine_grained_labels
        self._max_num_user_images = max_num_user_images

        with h5py.File(self._hdf5_path, "r", swmr=True) as h5:
            self._length = len(h5[f"/{self._partition}/{self._user_id}/image_ids"])

    def __len__(self) -> int:
        """Length of dataset."""
        return min(self._length, self._max_num_user_images)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get datapoint number `index` from user."""
        with h5py.File(self._hdf5_path, "r", swmr=True) as h5:
            input = np.array(h5[f"/{self._partition}/{self._user_id}/images"][0])
            target = get_multi_hot_target_one_sample(
                index,
                self._num_classes,
                h5,
                self._partition,
                self._user_id,
                self._use_fine_grained_labels,
            )
            input = torch.as_tensor(input)
            target = torch.as_tensor(target)
            return input, target
