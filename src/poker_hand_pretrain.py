import warnings

from pytorch_tabnet.pretraining import TabNetPretrainer
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

    X_train, _ = dataset["train_unlabeled"]
    X_test, y_test = dataset["test"]
    X_valid, y_valid = dataset["valid"]

    # pretrainer
    pretrained_model = TabNetPretrainer(
        cat_idxs=cat_idxs,
        cat_dims=cat_dims,
        cat_emb_dim=3,
        seed=seed,
        n_a=16,
        n_d=16,
        momentum=0.95,
        n_steps=4,
        gamma=1.5,
        n_shared=2,
        optimizer_fn=Adam,
        optimizer_params={"lr": 1e-2},
        scheduler_fn=ExponentialLR,
        scheduler_params={"gamma": 0.95},
    )

    # pretraining
    pretrained_model.fit(
        X_train=X_train,
        eval_set=[X_valid],
        eval_name=["valid"],
        pretraining_ratio=0.8,
        batch_size=4096,
        virtual_batch_size=1024,
        max_epochs=1000,
        patience=30,
    )

    # save pretrained model
    pretrained_model.save_model("./saved_models/tabnet/poker_hand_pretrained_1000")
