import warnings

from pytorch_tabnet.pretraining import TabNetPretrainer
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.metrics import accuracy_score

from datasets import get_dataset


if __name__ == "__main__":
    warnings.simplefilter("ignore")

    seed = 100

    # dataset
    dataset, cat_idxs, cat_dims = get_dataset(
        file_path="./data/uci_income/adult.csv",
        target="salary",
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
    )

    pretrained_model.fit(X_train=X_u_train, eval_set=[X_valid], pretraining_ratio=0.8)

    clf = TabNetClassifier(
        cat_idxs=cat_idxs, cat_dims=cat_dims, cat_emb_dim=3, seed=seed
    )

    clf.fit(
        X_train=X_l_train,
        y_train=y_l_train,
        eval_set=[(X_l_train, y_l_train), (X_valid, y_valid)],
        eval_name=["train", "test"],
        eval_metric=["accuracy"],
        from_unsupervised=pretrained_model,
    )

    pred_test = clf.predict(X_test)
    test_acc = accuracy_score(y_pred=pred_test, y_true=y_test)

    print(f"BEST VALID SCORE : {clf.best_cost}")
    print(f"FINAL TEST SCORE : {test_acc}")
