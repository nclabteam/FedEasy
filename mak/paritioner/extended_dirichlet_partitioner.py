import warnings
from typing import Any, Literal, Optional, Union

import datasets
import numpy as np
from flwr_datasets.common.typing import NDArray, NDArrayFloat
from flwr_datasets.partitioner.partitioner import Partitioner


class ExtendedDirichletPartitioner(Partitioner):
    def __init__(  # pylint: disable=R0913
        self,
        num_partitions: int,
        partition_by: str,
        alpha: Union[int, float, list[float], NDArrayFloat],
        num_classes_per_partition: int,
        class_assignment_mode: Literal[
            "random", "deterministic", "first-deterministic"
        ] = "random",
        min_partition_size: int = 10,
        self_balancing: bool = False,
        shuffle: bool = True,
        seed: Optional[int] = 42,
    ) -> None:
        super().__init__()
        # Attributes based on the constructor
        self._num_partitions = num_partitions
        self._check_num_partitions_greater_than_zero()
        self._alpha: NDArrayFloat = self._initialize_alpha(alpha)
        self._partition_by = partition_by
        self._min_partition_size: int = min_partition_size
        self._num_classes_per_partition = num_classes_per_partition
        self._class_assignment_mode = class_assignment_mode
        self._self_balancing = self_balancing
        self._shuffle = shuffle
        self._seed = seed
        self._rng = np.random.default_rng(seed=self._seed)  # NumPy random generator

        # Utility attributes
        # The attributes below are determined during the first call to load_partition
        self._avg_num_of_samples_per_partition: Optional[float] = None
        # self._unique_classes: Optional[Union[list[int], list[str]]] = None
        self._partition_id_to_indices: dict[int, list[int]] = {}
        self._partition_id_to_unique_labels: dict[int, list[Any]] = {
            pid: [] for pid in range(self._num_partitions)
        }
        self._unique_labels: list[Any] = []
        # Count in how many partitions the label is used
        self._unique_label_to_times_used_counter: dict[Any, int] = {}
        self._partition_id_to_indices_determined = False

    def load_partition(self, partition_id: int) -> datasets.Dataset:
        """Load a partition based on the partition index.

        Parameters
        ----------
        partition_id : int
            The index that corresponds to the requested partition.

        Returns
        -------
        dataset_partition : Dataset
            Single partition of a dataset.
        """
        # The partitioning is done lazily - only when the first partition is
        # requested. Only the first call creates the indices assignments for all the
        # partition indices.
        self._check_num_partitions_correctness_if_needed()
        self._determine_partition_id_to_indices_if_needed()
        return self.dataset.select(self._partition_id_to_indices[partition_id])

    @property
    def num_partitions(self) -> int:
        """Total number of partitions."""
        self._check_num_partitions_correctness_if_needed()
        self._determine_partition_id_to_indices_if_needed()
        return self._num_partitions

    def _initialize_alpha(
        self, alpha: Union[int, float, list[float], NDArrayFloat]
    ) -> NDArrayFloat:
        """Convert alpha to the used format in the code a NDArrayFloat.

        The alpha can be provided in constructor can be in different format for user
        convenience. The format into which it's transformed here is used throughout the
        code for computation.

        Parameters
        ----------
            alpha : Union[int, float, List[float], NDArrayFloat]
                Concentration parameter to the Dirichlet distribution

        Returns
        -------
        alpha : NDArrayFloat
            Concentration parameter in a format ready to used in computation.
        """
        if isinstance(alpha, int):
            alpha = np.array([float(alpha)], dtype=float).repeat(self._num_partitions)
        elif isinstance(alpha, float):
            alpha = np.array([alpha], dtype=float).repeat(self._num_partitions)
        elif isinstance(alpha, list):
            if len(alpha) != self._num_partitions:
                raise ValueError(
                    "If passing alpha as a List, it needs to be of length of equal to "
                    "num_partitions."
                )
            alpha = np.asarray(alpha)
        elif isinstance(alpha, np.ndarray):
            # pylint: disable=R1720
            if alpha.ndim == 1 and alpha.shape[0] != self._num_partitions:
                raise ValueError(
                    "If passing alpha as an NDArray, its length needs to be of length "
                    "equal to num_partitions."
                )
            elif alpha.ndim == 2:
                alpha = alpha.flatten()
                if alpha.shape[0] != self._num_partitions:
                    raise ValueError(
                        "If passing alpha as an NDArray, its size needs to be of length"
                        " equal to num_partitions."
                    )
        else:
            raise ValueError("The given alpha format is not supported.")
        if not (alpha > 0).all():
            raise ValueError(
                f"Alpha values should be strictly greater than zero. "
                f"Instead it'd be converted to {alpha}"
            )
        return alpha

    def _determine_partition_id_to_indices_if_needed(
        self,
    ) -> None:
        """Create an assignment of indices to the partition indices."""
        if self._partition_id_to_indices_determined:
            return
        self._determine_partition_id_to_unique_labels()
        assert self._unique_labels is not None
        self._count_partitions_having_each_unique_label()
        self._avg_num_of_samples_per_partition = (
            self.dataset.num_rows / self._num_partitions
        )
        # Change labels list data type to numpy
        labels = np.asarray(self.dataset[self._partition_by])
        self._check_correctness_of_unique_label_to_times_used_counter(labels)
        # Repeat the sampling procedure based on the Dirichlet distribution until the
        # min_partition_size is reached.
        sampling_try = 0
        while True:
            # Prepare data structure to store indices assigned to partition ids
            for partition_id in range(self._num_partitions):
                self._partition_id_to_indices[partition_id] = []

            unused_labels = []
            for unique_label in self._unique_labels:
                if self._unique_label_to_times_used_counter[unique_label] == 0:
                    unused_labels.append(unique_label)
                    continue
                # Get the indices in the original dataset where the y == unique_label
                unique_label_to_indices = np.where(labels == unique_label)[0]

                # Iterated over all unique labels (they are not necessarily of type int)
                # for k in self._unique_labels:
                #     # Access all the indices associated with class k
                #     unique_label_to_indices = np.nonzero(labels == k)[0]
                # Determine division (the fractions) of the data representing class k
                # among the partitions
                class_k_division_proportions = self._rng.dirichlet(self._alpha)
                nid_to_proportion_of_k_samples = {}
                for nid in range(self._num_partitions):
                    nid_to_proportion_of_k_samples[nid] = class_k_division_proportions[
                        nid
                    ]
                # Balancing (not mentioned in the paper but implemented)
                # Do not assign additional samples to the partition if it already has
                # more than the average numbers of samples per partition. Note that it
                # might especially affect classes that are later in the order. This is
                # the reason for more sparse division that the alpha might suggest.
                for nid in nid_to_proportion_of_k_samples.copy():
                    if (
                        unique_label
                        not in self._partition_id_to_unique_labels[nid]
                        # or len(self._partition_id_to_indices[nid])
                        #     > self._avg_num_of_samples_per_partition
                    ):
                        nid_to_proportion_of_k_samples[nid] = 0

                # Normalize the proportions such that they sum up to 1
                sum_proportions = sum(nid_to_proportion_of_k_samples.values())
                for nid, prop in nid_to_proportion_of_k_samples.copy().items():
                    nid_to_proportion_of_k_samples[nid] = prop / (sum_proportions)

                # Determine the split indices
                cumsum_division_fractions = np.cumsum(
                    list(nid_to_proportion_of_k_samples.values())
                )
                cumsum_division_numbers = cumsum_division_fractions * len(
                    unique_label_to_indices
                )
                # [:-1] is because the np.split requires the division indices but the
                # last element represents the sum = total number of samples
                indices_on_which_split = cumsum_division_numbers.astype(int)[:-1]

                split_indices = np.split(
                    unique_label_to_indices, indices_on_which_split
                )
                stored_nid = None
                for nid in nid_to_proportion_of_k_samples.copy():
                    if (
                        unique_label in self._partition_id_to_unique_labels[nid]
                        and len(split_indices[nid]) > 0
                    ):
                        stored_nid = nid
                    if (
                        unique_label not in self._partition_id_to_unique_labels[nid]
                        and len(split_indices[nid]) > 0
                    ):
                        split_indices[stored_nid] = np.append(
                            split_indices[stored_nid], split_indices[nid]
                        )
                        split_indices[nid] = np.delete(
                            split_indices[nid],
                            [i for i in range(len(split_indices[nid]))],
                        )
                # Append new indices (coming from class k) to the existing indices
                for nid, indices in self._partition_id_to_indices.items():
                    indices.extend(split_indices[nid].tolist())
                # if unique_label == 0: print(f"{unique_label}: {split_indices}")
            # Determine if the indices assignment meets the min_partition_size
            # If it does not mean the requirement repeat the Dirichlet sampling process
            # Otherwise break the while loop
            min_sample_size_on_client = min(
                len(indices) for indices in self._partition_id_to_indices.values()
            )
            if min_sample_size_on_client >= self._min_partition_size:
                break
            sample_sizes = [
                len(indices) for indices in self._partition_id_to_indices.values()
            ]
            alpha_not_met = [
                self._alpha[i]
                for i, ss in enumerate(sample_sizes)
                if ss == min(sample_sizes)
            ]
            mssg_list_alphas = (
                (
                    "Generating partitions by sampling from a list of very wide range "
                    "of alpha values can be hard to achieve. Try reducing the range "
                    f"between maximum ({max(self._alpha)}) and minimum alpha "
                    f"({min(self._alpha)}) values or increasing all the values."
                )
                if len(self._alpha.flatten().tolist()) > 0
                else ""
            )
            if len(unused_labels) >= 1:
                warnings.warn(
                    f"Classes: {unused_labels} will NOT be used due to the chosen "
                    f"configuration. If it is undesired behavior consider setting"
                    f" 'first_class_deterministic_assignment=True' which in case when"
                    f" the number of classes is smaller than the number of partitions will "
                    f"utilize all the classes for the created partitions.",
                    stacklevel=1,
                )

            warnings.warn(
                f"The specified min_partition_size ({self._min_partition_size}) was "
                f"not satisfied for alpha ({alpha_not_met}) after "
                f"{sampling_try} attempts at sampling from the Dirichlet "
                f"distribution. The probability sampling from the Dirichlet "
                f"distribution will be repeated. Note: This is not a desired "
                f"behavior. It is recommended to adjust the alpha or "
                f"min_partition_size instead. {mssg_list_alphas}",
                stacklevel=1,
            )
            if sampling_try == 10:
                raise ValueError(
                    "The max number of attempts (10) was reached. "
                    "Please update the values of alpha and try again."
                )
            sampling_try += 1

        # Shuffle the indices not to have the datasets with labels in sequences like
        # [00000, 11111, ...]) if the shuffle is True
        if self._shuffle:
            for indices in self._partition_id_to_indices.values():
                # In place shuffling
                self._rng.shuffle(indices)
        # self._partition_id_to_indices = partition_id_to_indices
        self._partition_id_to_indices_determined = True

    def _check_num_partitions_correctness_if_needed(self) -> None:
        """Test num_partitions when the dataset is given (in load_partition)."""
        if not self._partition_id_to_indices_determined:
            if self._num_partitions > self.dataset.num_rows:
                raise ValueError(
                    "The number of partitions needs to be smaller than the number of "
                    "samples in the dataset."
                )

    def _check_num_partitions_greater_than_zero(self) -> None:
        """Test num_partition left sides correctness."""
        if not self._num_partitions > 0:
            raise ValueError("The number of partitions needs to be greater than zero.")

    def _determine_partition_id_to_unique_labels(self) -> None:
        """Determine the assignment of unique labels to the partitions."""
        self._unique_labels = sorted(self.dataset.unique(self._partition_by))
        num_unique_classes = len(self._unique_labels)

        if self._num_classes_per_partition > num_unique_classes:
            raise ValueError(
                f"The specified `num_classes_per_partition`"
                f"={self._num_classes_per_partition} is greater than the number "
                f"of unique classes in the given dataset={num_unique_classes}. "
                f"Reduce the `num_classes_per_partition` or make use different dataset "
                f"to apply this partitioning."
            )
        if self._class_assignment_mode == "first-deterministic":
            # if self._first_class_deterministic_assignment:
            for partition_id in range(self._num_partitions):
                label = self._unique_labels[partition_id % num_unique_classes]
                self._partition_id_to_unique_labels[partition_id].append(label)

                while (
                    len(self._partition_id_to_unique_labels[partition_id])
                    < self._num_classes_per_partition
                ):
                    label = self._rng.choice(self._unique_labels, size=1)[0]
                    if label not in self._partition_id_to_unique_labels[partition_id]:
                        self._partition_id_to_unique_labels[partition_id].append(label)
        elif self._class_assignment_mode == "deterministic":
            for partition_id in range(self._num_partitions):
                labels = []
                for i in range(self._num_classes_per_partition):
                    label = self._unique_labels[
                        (partition_id + i) % len(self._unique_labels)
                    ]
                    labels.append(label)
                self._partition_id_to_unique_labels[partition_id] = labels
        elif self._class_assignment_mode == "random":
            for partition_id in range(self._num_partitions):
                labels = self._rng.choice(
                    self._unique_labels,
                    size=self._num_classes_per_partition,
                    replace=False,
                ).tolist()
                self._partition_id_to_unique_labels[partition_id] = labels
        else:
            raise ValueError(
                f"The supported class_assignment_mode are: 'random', 'deterministic', "
                f"'first-deterministic'. You provided: {self._class_assignment_mode}."
            )

    def _count_partitions_having_each_unique_label(self) -> None:
        """Count the number of partitions that have each unique label.

        This computation is based on the assigment of the label to the partition_id in
        the `_determine_partition_id_to_unique_labels` method.
        Given:
        * partition 0 has only labels: 0,1 (not necessarily just two samples it can have
          many samples but either from 0 or 1)
        *  partition 1 has only labels: 1, 2 (same count note as above)
        * and there are only two partitions then the following will be computed:
        {
          0: 1,
          1: 2,
          2: 1
        }
        """
        for unique_label in self._unique_labels:
            self._unique_label_to_times_used_counter[unique_label] = 0
        for unique_labels in self._partition_id_to_unique_labels.values():
            for unique_label in unique_labels:
                self._unique_label_to_times_used_counter[unique_label] += 1

    def _check_correctness_of_unique_label_to_times_used_counter(
        self, labels: NDArray
    ) -> None:
        """Check if partitioning is possible given the presence requirements.

        The number of times the label can be used must be smaller or equal to the number
        of times that the label is present in the dataset.
        """
        for unique_label in self._unique_labels:
            num_unique = np.sum(labels == unique_label)
            if self._unique_label_to_times_used_counter[unique_label] > num_unique:
                raise ValueError(
                    f"Label: {unique_label} is needed to be assigned to more "
                    f"partitions "
                    f"({self._unique_label_to_times_used_counter[unique_label]})"
                    f" than there are samples (corresponding to this label) in the "
                    f"dataset ({num_unique}). Please decrease the `num_partitions`, "
                    f"`num_classes_per_partition` to avoid this situation, "
                    f"or try `class_assigment_mode='deterministic'` to create a more "
                    f"even distribution of classes along the partitions. "
                    f"Alternatively use a different dataset if you can not adjust"
                    f" the any of these parameters."
                )
