import warnings

import numpy as np
from torch import nn, Tensor
import easyfl
from easyfl.datasets import FederatedTensorDataset

from utils import get_dataset


class MLP(nn.Module):
    def __init__(self, feature_num: int, class_num: int) -> None:
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(feature_num, feature_num),
            nn.ReLU(),
            nn.Linear(feature_num, class_num),
        )

    def forward(self, X: Tensor) -> Tensor:
        logits = self.mlp(X)
        return logits


if __name__ == "__main__":
    warnings.simplefilter("ignore")

    config = easyfl.load_config("./config/main.yaml")

    dataset, _, _ = get_dataset(
        file_path=config.data.file_path,
        target=config.data.target,
        labeled_size=config.data.labeled_size,
        train_size=config.data.train_size,
    )

    X_train, y_train = dataset["train_labeled"]
    X_test, y_test = dataset["test"]

    feature_num = X_train.shape[1]

    train_data = FederatedTensorDataset(
        data={"x": X_train.astype(dtype=np.float32), "y": y_train},
        num_of_clients=config.data.num_of_clients,
    )
    test_data = FederatedTensorDataset(
        data={"x": X_test.astype(dtype=np.float32), "y": y_test},
        num_of_clients=config.data.num_of_clients,
    )

    easyfl.register_dataset(train_data, test_data)
    easyfl.register_model(model=MLP(feature_num=feature_num, class_num=2))

    easyfl.init(config)

    easyfl.run()
