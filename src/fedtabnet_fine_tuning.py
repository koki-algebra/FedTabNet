import warnings

import easyfl
from easyfl.datasets import FederatedTensorDataset
from pytorch_tabnet.tab_network import TabNet, TabNetPretraining
from pytorch_tabnet.multiclass_utils import infer_output_dim

from utils import get_dataset
from tabnet import load_weights_from_pretrained, Client, Server


# FedTabNet の Fine-tuning を行う
if __name__ == "__main__":
    warnings.simplefilter("ignore")

    # 設定ファイルのロード
    config = easyfl.load_config("./config/poker_fine_tuning.yaml")

    # データセットのロード
    dataset, cat_idxs, cat_dims = get_dataset(
        file_path=config.data.file_path,
        target=config.data.target,
        labeled_size=config.data.labeled_size,
        train_size=config.data.train_size,
        seed=config.seed,
    )

    _, y_u_train = dataset["train_unlabeled"]
    X_train, y_train = dataset["train_labeled"]
    X_test, y_test = dataset["test"]

    # Federated Learning 用のデータセットを定義
    train_data = FederatedTensorDataset(
        data={"x": X_train, "y": y_train}, num_of_clients=config.data.num_of_clients
    )
    test_data = FederatedTensorDataset(
        data={"x": X_test, "y": y_test}, num_of_clients=config.data.num_of_clients
    )

    # クラス数を推論
    input_dim = X_train.shape[1]
    output_dim, _ = infer_output_dim(y_u_train)

    # 事前学習済モデルをロード
    pretrain_config = easyfl.load_config("./config/poker_pretrain.yaml")
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

    # Fine-tuning 用のモデルを定義
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

    load_weights_from_pretrained(
        model,
        pretarined_model,
        file_path="./saved_models/pretrain_global_model_r_4.pth",
    )

    easyfl.register_dataset(train_data=train_data, test_data=test_data)
    easyfl.register_model(model)
    easyfl.register_client(client=Client)
    easyfl.register_server(server=Server)

    easyfl.init(config)

    easyfl.run()
