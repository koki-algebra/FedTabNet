import logging
import math
from typing import Dict

import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from easyfl.datasets import FederatedDataset
from easyfl.datasets.dataset import default_process_x, default_process_y
from easyfl.datasets.simulation import data_simulation, SIMULATE_IID

logger = logging.getLogger(__name__)


class FederatedSemiSLDataset(FederatedDataset):
    def __init__(
        self,
        labeled_data: Dict[str, np.ndarray],
        unlabeled_data: Dict[str, np.ndarray],
        process_x=default_process_x,
        process_y=default_process_x,
        simulated=False,
        do_simulate=True,
        num_of_clients=10,
        simulation_method=SIMULATE_IID,
        weights=None,
        alpha=0.5,
        min_size=10,
        class_per_client=1,
    ):
        super().__init__()
        self.unlabeled_division = math.floor(
            unlabeled_data["x"].shape[0] / labeled_data["x"].shape[0]
        )

        self.simulated = simulated
        self.labeled_data = labeled_data
        self.unlabeled_data = unlabeled_data
        self._validate_data(labeled_data)
        self._validate_data(unlabeled_data)
        self.process_x = process_x
        self.process_y = process_y
        if simulated:
            self._users = sorted(list(self.labeled_data.keys()))

        elif do_simulate:
            # For simulation method provided, we support testing in server for now
            # TODO: support simulation for test data => test in clients
            self.simulation(
                num_of_clients,
                simulation_method,
                weights,
                alpha,
                min_size,
                class_per_client,
            )

    def simulation(
        self,
        num_of_clients,
        niid=SIMULATE_IID,
        weights=None,
        alpha=0.5,
        min_size=10,
        class_per_client=1,
    ):
        if self.simulated:
            logger.warning(
                "The dataset is already simulated, the simulation would not proceed."
            )
            return
        self._users, self.labeled_data = data_simulation(
            data_x=self.labeled_data["x"],
            data_y=self.labeled_data["y"],
            num_of_clients=num_of_clients,
            data_distribution=niid,
            weights=weights,
            alpha=alpha,
            min_size=min_size,
            class_per_client=class_per_client,
        )
        _, self.unlabeled_data = data_simulation(
            data_x=self.unlabeled_data["x"],
            data_y=self.unlabeled_data["y"],
            num_of_clients=num_of_clients,
            data_distribution=niid,
            weights=weights,
            alpha=alpha,
            min_size=min_size,
            class_per_client=class_per_client,
        )
        self.simulated = True

    def loader(
        self,
        batch_size,
        client_id=None,
        shuffle=True,
        seed=0,
        drop_last=False,
    ):
        if client_id is None:
            labeled_data = self.labeled_data
            unlabeled_data = self.unlabeled_data
        else:
            labeled_data = self.labeled_data[client_id]
            unlabeled_data = self.unlabeled_data[client_id]

        labeled_data_x = labeled_data["x"]
        labeled_data_y = labeled_data["y"]
        unlabeled_data_x = unlabeled_data["x"]
        unlabeled_data_y = unlabeled_data["y"]

        labeled_data_x = self._input_process(labeled_data_x)
        unlabeled_data_x = self._input_process(unlabeled_data_x)
        labeled_data_y = self._label_process(labeled_data_y)
        unlabeled_data_y = self._label_process(unlabeled_data_y)

        labeled_dataset = TensorDataset(labeled_data_x, labeled_data_y)
        unlabeled_dataset = TensorDataset(unlabeled_data_x, unlabeled_data_y)

        loader = {
            "labeled": DataLoader(
                dataset=labeled_dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                drop_last=drop_last,
            ),
            "unlabeled": DataLoader(
                dataset=unlabeled_dataset,
                batch_size=batch_size * self.unlabeled_division,
                shuffle=shuffle,
                drop_last=drop_last,
            ),
        }

        return loader

    @property
    def users(self):
        return self._users

    @users.setter
    def users(self, value):
        self._users = value

    def size(self, cid=None):
        if cid is not None:
            return len(self.labeled_data[cid]["y"])
        else:
            return len(self.labeled_data["y"])

    def total_size(self):
        if "y" in self.labeled_data:
            return len(self.labeled_data["y"])
        else:
            return sum([len(self.labeled_data[i]["y"]) for i in self.labeled_data])

    def _input_process(self, sample):
        if self.process_x is not None:
            sample = self.process_x(sample)
        return sample

    def _label_process(self, label):
        if self.process_y is not None:
            label = self.process_y(label)
        return label

    def _validate_data(self, data):
        if self.simulated:
            for i in data:
                assert len(data[i]["x"]) == len(data[i]["y"])
        else:
            assert len(data["x"]) == len(data["y"])
