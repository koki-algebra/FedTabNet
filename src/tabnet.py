import warnings

from pytorch_tabnet.pretraining import TabNetPretrainer
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.metrics import accuracy_score
from torch.optim.lr_scheduler import ExponentialLR
from torch.optim import Adam

from utils import get_dataset


if __name__ == "__main__":
    warnings.simplefilter("ignore")

    # 乱数シード
    seed = 0

    """
    データセットのロード
    この例では
    訓練データ:テストデータ = 8:2
    訓練データ中のラベル付きデータの割合 10%
    テストデータ中の検証データの割合 20%
    としている
    """
    dataset, cat_idxs, cat_dims = get_dataset(
        file_path="./data/covtype/covtype.zip",    # データセットのパスを指定
        target="Cover_Type",                       # ラベルのカラム名を指定
        labeled_size=0.1,                          # 訓練データ中のラベル付きデータの割合を指定
        train_size=0.8,                            # 訓練データの割合を指定
        valid_size=0.2,                            # 検証データの割合を指定
        seed=seed,                                 # シードの指定
    )

    X_u_train, _ = dataset["train_unlabeled"]          # ラベルなしデータ
    X_l_train, y_l_train = dataset["train_labeled"]    # ラベル付きデータ
    X_test, y_test = dataset["test"]                   # テストデータ
    X_valid, y_valid = dataset["valid"]                # 検証データ

    # 事前学習モデルを定義
    # 各種ハイパーパラメータを設定 (下記参照)
    # https://github.com/dreamquark-ai/tabnet
    pretrained_model = TabNetPretrainer(
        cat_idxs=cat_idxs,
        cat_dims=cat_dims,
        cat_emb_dim=3,
        seed=seed,
        n_a=64,
        n_d=64,
        momentum=0.7,
        n_steps=5,
        gamma=1.5,
        n_shared=2,
        optimizer_fn=Adam,
        optimizer_params={"lr": 2e-2},
        scheduler_fn=ExponentialLR,
        scheduler_params={"gamma": 0.95},
    )

    # 事前学習
    pretrained_model.fit(
        X_train=X_u_train,
        eval_set=[X_valid],
        eval_name=["valid"],
        pretraining_ratio=0.8,
        batch_size=2**14,
        virtual_batch_size=512,
        max_epochs=1000,
        patience=30,
    )

    # 事前学習済モデルを保存
    pretrained_model.save_model("./saved_models/tabnet/covtype_pretrained")

    # fine-tuning 用の classifier を定義
    clf = TabNetClassifier(
        seed=seed,
        optimizer_fn=Adam,
        optimizer_params={"lr": 2e-2},
        scheduler_fn=ExponentialLR,
        scheduler_params={"gamma": 0.95},
    )

    # fine-tuning
    clf.fit(
        X_train=X_l_train,
        y_train=y_l_train,
        eval_set=[(X_l_train, y_l_train), (X_valid, y_valid)],
        eval_name=["train", "valid"],
        eval_metric=["accuracy"],
        from_unsupervised=pretrained_model,
        batch_size=1024,
        virtual_batch_size=512,
        max_epochs=1000,
        patience=30,
    )

    # テストデータを予測
    pred_test = clf.predict(X_test)
    test_acc = accuracy_score(y_pred=pred_test, y_true=y_test)

    print(f"BEST VALID SCORE : {clf.best_cost}")
    print(f"FINAL TEST SCORE : {test_acc}")
