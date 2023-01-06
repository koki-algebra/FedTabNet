import warnings

from pytorch_tabnet.pretraining import TabNetPretrainer
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.metrics import accuracy_score
from torch.optim.lr_scheduler import ExponentialLR
from torch.optim import Adam

from utils import get_dataset


if __name__ == "__main__":
    warnings.simplefilter("ignore")

    seed = 1000

    # dataset
    dataset, cat_idxs, cat_dims = get_dataset(
        file_path="./data/poker_hand/poker_hand.csv",
        target="CLASS",
        labeled_size=0.1,
        train_size=0.8,
        valid_size=0.2,
        seed=seed,
    )

    X_train, y_train = dataset["train_labeled"]
    X_test, y_test = dataset["test"]
    X_valid, y_valid = dataset["valid"]

    # load pretrained model
    pretrained_model = TabNetPretrainer()
    pretrained_model.load_model("./saved_models/tabnet/poker_hand_pretrained_1000.zip")

    # classifier
    clf = TabNetClassifier(
        seed=seed,
        optimizer_fn=Adam,
        optimizer_params={"lr": 2e-2},
        scheduler_fn=ExponentialLR,
        scheduler_params={"gamma": 0.95},
    )

    # fine-tuning
    clf.fit(
        X_train=X_train,
        y_train=y_train,
        eval_set=[(X_train, y_train), (X_valid, y_valid)],
        eval_name=["train", "valid"],
        eval_metric=["accuracy"],
        from_unsupervised=pretrained_model,
        batch_size=1024,
        virtual_batch_size=512,
        max_epochs=1000,
        patience=30,
    )

    # test
    pred_test = clf.predict(X_test)
    test_acc = accuracy_score(y_pred=pred_test, y_true=y_test)

    print(f"BEST VALID SCORE : {clf.best_cost}")
    print(f"FINAL TEST SCORE : {test_acc}")
