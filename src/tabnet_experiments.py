import warnings

from pytorch_tabnet.pretraining import TabNetPretrainer
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.metrics import accuracy_score
from torch.optim.lr_scheduler import ExponentialLR
from torch.optim import Adam

from utils import get_dataset


if __name__ == "__main__":
    warnings.simplefilter("ignore")

    seed = 100

    # dataset
    dataset, cat_idxs, cat_dims = get_dataset(
        file_path="./data/covtype/covtype.zip",
        target="Cover_Type",
        labeled_size=0.1,
        train_size=0.8,
        valid_size=0.2,
        seed=seed,
    )

    X_l_train, y_l_train = dataset["train_labeled"]
    X_u_train, _ = dataset["train_unlabeled"]
    X_test, y_test = dataset["test"]
    X_valid, y_valid = dataset["valid"]

    # pretrainer
    pretrained_model = TabNetPretrainer(
        cat_idxs=cat_idxs,
        cat_dims=cat_dims,
        cat_emb_dim=3,
        seed=seed,
        n_a=64,
        n_d=64,
        lambda_sparse=1e-4,
        momentum=0.7,
        n_steps=5,
        gamma=1.5,
        n_shared=2,
        optimizer_fn=Adam,
        optimizer_params={"lr": 2e-2},
        scheduler_fn=ExponentialLR,
        scheduler_params={"gamma": 0.95},
    )

    pretrained_model.fit(
        X_train=X_u_train,
        eval_set=[X_valid],
        pretraining_ratio=0.8,
        batch_size=16384,
        virtual_batch_size=512,
    )

    pretrained_model.save_model("./saved_models/tabnet/covtype.pth")

    clf = TabNetClassifier(
        cat_idxs=cat_idxs,
        cat_dims=cat_dims,
        cat_emb_dim=3,
        seed=seed,
        optimizer_fn=Adam,
        optimizer_params=dict(lr=2e-2),
        scheduler_fn=ExponentialLR,
        scheduler_params={"gamma": 0.95},
    )

    clf.fit(
        X_train=X_l_train,
        y_train=y_l_train,
        eval_set=[(X_l_train, y_l_train), (X_valid, y_valid)],
        eval_name=["train", "valid"],
        eval_metric=["accuracy"],
        from_unsupervised=pretrained_model,
        batch_size=1024,
        virtual_batch_size=512,
    )

    pred_test = clf.predict(X_test)
    test_acc = accuracy_score(y_pred=pred_test, y_true=y_test)

    print(f"BEST VALID SCORE : {clf.best_cost}")
    print(f"FINAL TEST SCORE : {test_acc}")
