import time
import logging

import torch
from torch import Tensor
from torch.nn.utils.clip_grad import clip_grad_norm
from easyfl.client.base import BaseClient

from utils import generate_pseudo_labels, pseudo_labeling_schedular

logger = logging.getLogger(__name__)


class PseudoLabelingClient(BaseClient):
    def __init__(
        self,
        cid,
        conf,
        train_data,
        test_data,
        device,
        sleep_time=0,
        is_remote=False,
        local_port=23000,
        server_addr="localhost:22999",
        tracker_addr="localhost:12666",
    ):
        super().__init__(
            cid,
            conf,
            train_data,
            test_data,
            device,
            sleep_time,
            is_remote,
            local_port,
            server_addr,
            tracker_addr,
        )

    def train(self, conf, device):
        torch.autograd.anomaly_mode.set_detect_anomaly(True)

        start_time = time.time()
        loss_fn, optimizer = self.pretrain_setup(conf, device)
        self.train_loss = []
        total_epoch = self.round_time * conf.local_epoch

        for i in range(conf.local_epoch):
            total_epoch += i

            # current epoch model for pseudo-labeling
            current_epoch_model: torch.nn.Module = self.model

            batch_loss = []
            labeled_loader = self.train_loader["labeled"]
            unlabeled_loader = self.train_loader["unlabeled"]

            for (X_l, y), (X_u, _) in zip(labeled_loader, unlabeled_loader):
                # Generate Pseudo-labels
                y_pseudo: Tensor = generate_pseudo_labels(current_epoch_model, X_u)

                X_l: Tensor = X_l.to(device).float()
                y: Tensor = y.to(device).long()
                X_u: Tensor = X_u.to(device).float()
                y_pseudo = y_pseudo.to(device).long()

                # predict labeled data
                labeled_output, M_loss = self.model(X_l)
                # predict unlabeled data
                unlabeled_output, _ = self.model(X_u)

                labeled_loss: Tensor = loss_fn(labeled_output, y)
                unlabeled_loss: Tensor = loss_fn(unlabeled_output, y_pseudo)

                # Add the overall sparsity loss
                alpha = pseudo_labeling_schedular(
                    t=total_epoch, T_1=conf.t_1, T_2=conf.t_2, alpha=conf.alpha
                )
                loss = (
                    labeled_loss - conf.lambda_sparse * M_loss + unlabeled_loss * alpha
                )

                # Perform backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                if conf.clip_value:
                    clip_grad_norm(self.model.parameters(), conf.clip_value)
                optimizer.step()

                # batch loss
                batch_loss.append(loss.item())

            current_epoch_loss = sum(batch_loss) / len(batch_loss)
            self.train_loss.append(float(current_epoch_loss))
            logger.info(
                "Client {}, local epoch: {}, loss: {}".format(
                    self.cid, i, current_epoch_loss
                )
            )

        self.train_time = time.time() - start_time
        logger.info("Client {}, Train Time: {}".format(self.cid, self.train_time))

    def test(self, conf, device):
        begin_test_time = time.time()
        self.model.eval()
        self.model.to(device)

        loss_fn = self.load_loss_fn(conf)

        if self.test_loader is None:
            self.test_loader = self.test_data.loader(
                conf.test_batch_size, self.cid, shuffle=False, seed=conf.seed
            )

        self.test_loss = 0
        correct = 0
        with torch.no_grad():
            for X, y in self.test_loader:
                X: Tensor = X.to(device).float()
                y: Tensor = y.to(device).long()

                # predict
                log_probs, _ = self.model(X)

                # loss
                loss = loss_fn(log_probs, y)
                _, y_pred = torch.max(log_probs, -1)
                correct += y_pred.eq(y.data.view_as(y_pred)).long().cpu().sum()
                self.test_loss += loss.item()

            test_size = self.test_data.size(self.cid)
            self.test_loss /= test_size
            self.test_accuracy = 100.0 * float(correct) / test_size

        logger.info(
            "Client {}, testing -- Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)".format(
                self.cid, self.test_loss, correct, test_size, self.test_accuracy
            )
        )

        self.test_time = time.time() - begin_test_time
        self.model = self.model.cpu()
