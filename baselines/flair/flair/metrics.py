"""Define utilities to modify torchvision model."""
# Code in this file is adapted from:
#
# https://github.com/apple/pfl-research/blob/main/benchmarks/model/pytorch/module_modification.py
# Copyright © 2023-2024 Apple Inc.
# Licensed under the Apache License, Version 2.0 (the "License").

# Disable some pylints to not have to modify the copied code.
# pylint: disable=invalid-name,protected-access
# pylint: inconsistent-return-statements,no-else-return
# ruff: noqa: D102
# ruff: noqa: D105
from enum import Enum
from typing import Dict, Optional, Protocol, Tuple, TypeVar, Union

import numpy as np

_SMALL_VALUE = 1e-7


class Summable(Protocol):
    """Summable metric values.

    Metric values are defined as a tuple of summable sufficient statistics `(num,
    denom)` that can be summed across users to accumulate the metric over all users.

    Then the final metric value can be obtained by `num/denom`.
    """

    def __add__(self, other: "Summable") -> "Summable":
        """Add interface."""
        ...


# Type variable bound to Summable protocol
TSummable = TypeVar("TSummable", bound=Summable)
Metric = Dict[str, Union[TSummable, Tuple[TSummable, TSummable]]]


def unsorted_segment_sum(data, segment_ids, num_segments):
    """Numpy implementation of unsorted segment sum in TensorFlow.

    Detailed at https://www.tensorflow.org/api_docs/
    python/tf/math/unsorted_segment_sum.
    """
    summed_segments = np.zeros(num_segments)
    np.add.at(summed_segments, segment_ids, data)
    return summed_segments


class BucketConfusionMatrix:
    """Confusion matrix for threshold buckets.

    Counts of true positives, false positives, true negatives and
    false negatives.

    :param true_positive:
        A numpy array in shape [num_thresholds, num_labels] representing count
        of true positive in each bucket for each label.
    :param false_positive:
        A numpy array in shape [num_thresholds, num_labels] representing count
        of false positive in each bucket for each label.
    :param true_negative:
        A numpy array in shape [num_thresholds, num_labels] representing count
        of true negative in each bucket for each label.
    :param false_negative:
        A numpy array in shape [num_thresholds, num_labels] representing count
        of false negative in each bucket for each label.
    """

    def __init__(
        self,
        true_positive: np.ndarray,
        false_positive: np.ndarray,
        true_negative: np.ndarray,
        false_negative: np.ndarray,
    ):
        shape = true_positive.shape
        assert all(
            arr.shape == shape
            for arr in [false_positive, true_negative, false_negative]
        )
        self._true_positive = true_positive
        self._false_positive = false_positive
        self._true_negative = true_negative
        self._false_negative = false_negative
        self._shape = shape

    @property
    def shape(self):
        return self._shape

    @classmethod
    def from_predictions(
        cls,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        num_thresholds: int,
        multi_label: bool,
    ):
        """Construct the confusion matrix based on models output prediction.

        Implementation follows TensorFlow Keras:
        https://github.com/keras-team/keras/blob/master/keras/utils/metrics_utils.py#L268.

        :param y_true:
            A numpy array in shape [num_data_points, num_labels] of binary
            classification labels
        :param y_pred:
            A numpy array in shape [num_data_points, num_labels] of binary
            classification scores
        :param num_thresholds:
            The number of thresholds to use when discretizing the scores.
            Values must be > 1.
        :param multi_label:
            boolean indicating whether multi-label data should be treated as
            multi-label or single-label. If False, predictions will be flatten
            and there is no distinction between different labels
            (i.e. similar to micro averaging metrics).

        :return:
            A ``BucketConfusionMatrix`` object
        """
        true_labels = y_true
        false_labels = 1 - y_true

        # Compute the bucket indices for each prediction value.
        # Since the predict value has to be greater than the thresholds,
        # eg, buckets like [0, 0.5], (0.5, 1], and 0.5 belongs to first bucket.
        # We have to use math.ceil(val) - 1 for the bucket.
        bucket_indices = np.ceil(y_pred * (num_thresholds - 1)) - 1
        bucket_indices = np.maximum(bucket_indices, 0).astype(int)

        def gather_bucket(label, bucket_index):
            return unsorted_segment_sum(
                data=label, segment_ids=bucket_index, num_segments=num_thresholds
            )

        if multi_label:
            true_labels, false_labels = true_labels.T, false_labels.T
            bucket_indices = bucket_indices.T
            tp_bucket_v_list = []
            fp_bucket_v_list = []
            for true_label, false_label, bucket_index in zip(
                true_labels, false_labels, bucket_indices
            ):
                tp_bucket_v_list.append(gather_bucket(true_label, bucket_index))
                fp_bucket_v_list.append(gather_bucket(false_label, bucket_index))
            tp_bucket_v = np.vstack(tp_bucket_v_list)
            fp_bucket_v = np.vstack(fp_bucket_v_list)
            tp = np.cumsum(tp_bucket_v[:, ::-1], axis=1)[:, ::-1].T
            fp = np.cumsum(fp_bucket_v[:, ::-1], axis=1)[:, ::-1].T
            total_true_labels = np.sum(true_labels, axis=1)[None, :]
            total_false_labels = np.sum(false_labels, axis=1)[None, :]
        else:
            tp_bucket_v = gather_bucket(true_labels, bucket_indices)
            fp_bucket_v = gather_bucket(false_labels, bucket_indices)
            tp = np.cumsum(tp_bucket_v[::-1])[::-1]
            fp = np.cumsum(fp_bucket_v[::-1])[::-1]
            total_true_labels = np.sum(true_labels)
            total_false_labels = np.sum(false_labels)

        tn = total_false_labels - fp
        fn = total_true_labels - tp
        return cls(tp, fp, tn, fn)

    def __add__(self, other):
        assert isinstance(other, BucketConfusionMatrix) and self._shape == other._shape
        return BucketConfusionMatrix(
            self._true_positive + other._true_positive,
            self._false_positive + other._false_positive,
            self._true_negative + other._true_negative,
            self._false_negative + other._false_negative,
        )

    def __eq__(self, other):
        assert isinstance(other, BucketConfusionMatrix)
        return all(
            np.array_equal(getattr(self, name), getattr(other, name))
            for name in [
                "_true_positive",
                "_false_positive",
                "_true_negative",
                "_false_negative",
            ]
        )

    @property
    def true_positive(self):
        return self._true_positive

    @property
    def false_positive(self):
        return self._false_positive

    @property
    def true_negative(self):
        return self._true_negative

    @property
    def false_negative(self):
        return self._false_negative

    @property
    def true_positive_rate(self):
        return np.nan_to_num(
            self._true_positive / (self._true_positive + self._false_negative)
        )

    @property
    def false_positive_rate(self):
        return np.nan_to_num(
            self._false_positive / (self._false_positive + self._true_negative)
        )

    @property
    def precision(self):
        return np.nan_to_num(
            self._true_positive / (self._true_positive + self._false_positive)
        )

    @property
    def recall(self):
        return self.true_positive_rate

    def from_vector(self, vector) -> "BucketConfusionMatrix":
        assert vector[0].shape == self._shape
        return BucketConfusionMatrix(*vector)

    def to_vector(self) -> np.ndarray:
        return np.array(
            [
                self._true_positive,
                self._false_positive,
                self._true_negative,
                self._false_negative,
            ],
            dtype=np.float32,
        )

    def __repr__(self):
        return f"ConfusionMatrix({self._shape})"


class AUCSummationMethod(Enum):
    """Type of AUC summation method.

    Contains the following values:

    * 'interpolation': Applies mid-point summation scheme for `ROC` curve. For
    `PR` curve, interpolates (true/false) positives but not the ratio that is
    precision (see Davis & Goadrich 2006 for details).
    * 'minoring': Applies left summation for increasing intervals and right
    summation for decreasing intervals.
    * 'majoring': Applies right summation for increasing intervals and left
    summation for decreasing intervals.
    """

    INTERPOLATION = "interpolation"
    MAJORING = "majoring"
    MINORING = "minoring"


class AUC(Summable):
    """AUC.

    Class for computing Area-Under-the-Curve metrics. Either
    a ``confusion_matrix`` or ``(y_true, y_pred)`` should be provided.

    :param confusion_matrix:
        A ``BucketConfusionMatrix`` object.
    :param y_true:
        A numpy array in shape [num_data_points, num_labels] of binary
        classification labels
    :param y_pred:
        A numpy array in shape [num_data_points, num_labels] of binary
        classification scores
    :param num_thresholds:
        The number of thresholds to use when discretizing the scores.
        Values must be > 1.
    :param multi_label:
        boolean indicating whether multi-label data should be treated as
        multi-label or single-label. If False, predictions will be flatten
        and there is no distinction between different labels
        (i.e. similar to micro averaging metrics).
    :param summation_method:
        Type of AUC summation method.
    """

    def __init__(
        self,
        confusion_matrix: Optional[BucketConfusionMatrix] = None,
        y_true: Optional[np.ndarray] = None,
        y_pred: Optional[np.ndarray] = None,
        multi_label: bool = False,
        num_thresholds: int = 200,
        summation_method: AUCSummationMethod = AUCSummationMethod.INTERPOLATION,
    ):
        if num_thresholds <= 1:
            raise ValueError(
                "Argument `num_thresholds` must be an integer > 1. "
                f"Received: num_thresholds={num_thresholds}"
            )

        # Otherwise, linearly interpolate (num_thresholds - 2) thresholds in
        # (0, 1).
        self._num_thresholds = num_thresholds
        thresholds = [
            (i + 1) * 1.0 / (num_thresholds - 1) for i in range(num_thresholds - 2)
        ]
        # Add an endpoint "threshold" below zero and above one for either
        # threshold method to account for floating point imprecisions.
        self._thresholds = np.array(
            [0.0 - _SMALL_VALUE, *thresholds, 1.0 + _SMALL_VALUE]
        )

        self._summation_method = summation_method
        self._multi_label = multi_label

        if confusion_matrix is None:
            assert y_true is not None and y_pred is not None
            if not self._multi_label:
                y_true = y_true.flatten()
                y_pred = y_pred.flatten()

            # build confusion matrix from model predicted scores
            confusion_matrix = BucketConfusionMatrix.from_predictions(
                y_true, y_pred, self._num_thresholds, self._multi_label
            )

        self._confusion_matrix = confusion_matrix

    def __eq__(self, other):
        assert isinstance(other, AUC)
        return (
            self._confusion_matrix == other._confusion_matrix
            and self._summation_method == other._summation_method
            and self._multi_label == other._multi_label
            and self._num_thresholds == other._num_thresholds
        )

    def __add__(self, other: Summable) -> Summable:
        assert isinstance(other, AUC)
        return self.__class__(
            self._confusion_matrix + other._confusion_matrix,
            num_thresholds=self._num_thresholds,
            summation_method=self._summation_method,
            multi_label=self._multi_label,
        )

    def to_vector(self) -> np.ndarray:
        return self._confusion_matrix.to_vector()

    def from_vector(self, vector: np.ndarray) -> "AUC":
        return self.__class__(
            self._confusion_matrix.from_vector(vector),
            num_thresholds=self._num_thresholds,
            summation_method=self._summation_method,
            multi_label=self._multi_label,
        )


class AveragedPrecision(AUC):
    """Averaged Precision.

    Averaged Precision summarizes a precision-recall curve as the weighted mean of
    precisions achieved at each threshold, with the increase in recall from the previous
    threshold used as the weight.

    More details in: https://scikit-
    learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html
    """

    @property
    def overall_value(self):
        # https://github.com/scikit-learn/scikit-learn/
        # blob/main/sklearn/metrics/_ranking.py#L202
        recall = self._confusion_matrix.recall
        precision = self._confusion_matrix.precision
        if self._multi_label:
            by_label_ap = -np.sum(
                np.diff(recall, axis=0) * np.array(precision)[:-1], axis=0
            )
            # Macro average of the label AUCs.
            return np.mean(by_label_ap)
        else:
            return -np.sum(np.diff(recall) * np.array(precision)[:-1])

    def __repr__(self):
        return "{} Averaged Precision: {:.3f}".format(
            "Macro" if self._multi_label else "Micro", self.overall_value
        )
