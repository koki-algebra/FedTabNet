import warnings

from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.metrics import accuracy_score
from torch.optim.lr_scheduler import ExponentialLR
from torch.optim import Adam

from utils import get_dataset


if __name__ == "__main__":
    warnings.simplefilter("ignore")

    seed = 5

    # dataset
    dataset, cat_idxs, cat_dims = get_dataset(
        file_path="./data/covtype/covtype.zip",
        target="Cover_Type",
        labeled_size=0.9999,
        train_size=0.8,
        valid_size=0.2,
        seed=seed,
    )

    X_train, y_train = dataset["train_labeled"]
    X_test, y_test = dataset["test"]
    X_valid, y_valid = dataset["valid"]

    # classifier
    clf = TabNetClassifier(
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

    # fitting
    clf.fit(
        X_train=X_train,
        y_train=y_train,
        eval_set=[(X_train, y_train), (X_valid, y_valid)],
        eval_name=["train", "valid"],
        eval_metric=["accuracy"],
        batch_size=2**14,
        virtual_batch_size=2**9,
        max_epochs=1000,
        patience=30,
    )

    # test
    pred_test = clf.predict(X_test)
    test_acc = accuracy_score(y_pred=pred_test, y_true=y_test)

    print(f"BEST VALID SCORE : {clf.best_cost}")
    print(f"FINAL TEST SCORE : {test_acc}")
