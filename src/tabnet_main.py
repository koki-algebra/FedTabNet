import warnings

import easyfl
from easyfl.datasets import FederatedTensorDataset

from datasets import get_dataset


if __name__ == "__main__":
    warnings.simplefilter("ignore")

    config = easyfl.load_config("./config/tabnet_main.yaml")

    dataset, cat_idxs, cat_dims = get_dataset(
        file_path=config.data.file_path,
        target=config.data.target,
        labeled_size=config.data.labeled_size,
        train_size=config.data.train_size,
        seed=config.seed,
    )

    X_train, y_train = dataset["train_labeled"]
    X_test, y_test = dataset["test"]

    train_data = FederatedTensorDataset(
        data={"x": X_train, "y": y_train}, num_of_clients=config.data.num_of_clients
    )
    test_data = FederatedTensorDataset(
        data={"x": X_test, "y": y_test}, num_of_clients=config.data.num_of_clients
    )

    easyfl.register_dataset(train_data=train_data, test_data=test_data)

    easyfl.init(config)
