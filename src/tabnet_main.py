import warnings

import easyfl
from easyfl.datasets import FederatedTensorDataset
from pytorch_tabnet.tab_network import TabNet
from pytorch_tabnet.multiclass_utils import infer_output_dim

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

    params = config.model_parameters
    output_dim, _ = infer_output_dim(y_train)

    easyfl.register_dataset(train_data=train_data, test_data=test_data)
    easyfl.register_model(
        model=TabNet(
            input_dim=X_train.shape[1],
            output_dim=output_dim,
            n_d=params.n_d,
            n_a=params.n_a,
            n_steps=params.n_steps,
            gamma=params.gamma,
            cat_idxs=cat_idxs,
            cat_dims=cat_dims,
            cat_emb_dim=params.cat_emb_dim,
            n_independent=params.n_independent,
            n_shared=params.n_shared,
            epsilon=params.epsilon,
            virtual_batch_size=params.virtual_batch_size,
            momentum=params.momentum,
            mask_type=params.mask_type,
        )
    )

    easyfl.init(config)
