import warnings

import easyfl
from easyfl.datasets import FederatedTensorDataset
from pytorch_tabnet.tab_network import TabNetPretraining

from utils import get_dataset
from tabnet import PretrainerClient, PretrainerServer


if __name__ == "__main__":
    warnings.simplefilter("ignore")

    config = easyfl.load_config("./config/income_pretrain.yaml")

    dataset, cat_idxs, cat_dims = get_dataset(
        file_path=config.data.file_path,
        target=config.data.target,
        labeled_size=config.data.labeled_size,
        train_size=config.data.train_size,
        seed=config.seed,
    )

    X_train, y_train = dataset["train_unlabeled"]

    train_data = FederatedTensorDataset(
        data={"x": X_train, "y": y_train}, num_of_clients=config.data.num_of_clients
    )

    # model parameters
    params = config.model_parameters

    easyfl.register_dataset(train_data=train_data, test_data=None)
    easyfl.register_model(
        model=TabNetPretraining(
            input_dim=X_train.shape[1],
            pretraining_ratio=params.pretraining_ratio,
            n_d=params.n_d,
            n_a=params.n_a,
            n_steps=params.n_steps,
            gamma=params.gamma,
            cat_idxs=cat_idxs,
            cat_dims=cat_dims,
            cat_emb_dim=params.cat_emb_dim,
            n_independent=params.n_independent,
            n_shared=params.n_shared,
            virtual_batch_size=params.virtual_batch_size,
            momentum=params.momentum,
            mask_type=params.mask_type,
            n_shared_decoder=params.n_shared_decoder,
            n_indep_decoder=params.n_indep_decoder,
        )
    )
    easyfl.register_server(server=PretrainerServer)
    easyfl.register_client(client=PretrainerClient)

    easyfl.init(config)

    easyfl.run()
