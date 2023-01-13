import warnings

import easyfl
from easyfl.datasets import FederatedTensorDataset
from pytorch_tabnet.tab_network import TabNet, TabNetPretraining
from pytorch_tabnet.multiclass_utils import infer_output_dim

from utils import get_dataset, FederatedSemiSLDataset
from tabnet import load_weights_from_pretrained, PseudoLabelingClient, Server


if __name__ == "__main__":
    warnings.simplefilter("ignore")

    # load config
    config = easyfl.load_config("./config/proposal/poker_fine_tuning.yaml")

    # dataset
    dataset, cat_idxs, cat_dims = get_dataset(
        file_path=config.data.file_path,
        target=config.data.target,
        labeled_size=config.data.labeled_size,
        train_size=config.data.train_size,
        seed=config.seed,
    )

    X_l_train, y_l_train = dataset["train_labeled"]
    X_u_train, y_u_train = dataset["train_unlabeled"]
    X_test, y_test = dataset["test"]

    train_data = FederatedSemiSLDataset(
        labeled_data={"x": X_l_train, "y": y_l_train},
        unlabeled_data={"x": X_u_train, "y": y_u_train},
        num_of_clients=config.data.num_of_clients,
    )
    test_data = FederatedTensorDataset(
        data={"x": X_test, "y": y_test}, num_of_clients=config.data.num_of_clients
    )

    input_dim = X_u_train.shape[1]
    output_dim, _ = infer_output_dim(y_u_train)

    # pretrained model
    pretrain_config = easyfl.load_config("./config/proposal/poker_pretrain.yaml")
    pretrain_params = pretrain_config.model_parameters
    pretarined_model = TabNetPretraining(
        input_dim=input_dim,
        pretraining_ratio=pretrain_params.pretraining_ratio,
        n_d=pretrain_params.n_d,
        n_a=pretrain_params.n_a,
        n_steps=pretrain_params.n_steps,
        gamma=pretrain_params.gamma,
        cat_idxs=cat_idxs,
        cat_dims=cat_dims,
        cat_emb_dim=pretrain_params.cat_emb_dim,
        n_independent=pretrain_params.n_independent,
        n_shared=pretrain_params.n_shared,
        virtual_batch_size=pretrain_params.virtual_batch_size,
        momentum=pretrain_params.momentum,
        mask_type=pretrain_params.mask_type,
        n_shared_decoder=pretrain_params.n_shared_decoder,
        n_indep_decoder=pretrain_params.n_indep_decoder,
    )

    # model
    params = config.model_parameters
    model = TabNet(
        input_dim=input_dim,
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
        virtual_batch_size=params.virtual_batch_size,
        momentum=params.momentum,
        mask_type=params.mask_type,
    )

    # load weights from pretrained model
    load_weights_from_pretrained(
        model,
        pretarined_model,
        file_path="./saved_models/pretrain_global_model_r_2.pth",
    )

    easyfl.register_dataset(train_data=train_data, test_data=test_data)
    easyfl.register_model(model)
    easyfl.register_client(client=PseudoLabelingClient)
    easyfl.register_server(server=Server)

    easyfl.init(config)

    easyfl.run()
