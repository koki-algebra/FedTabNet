import time
import logging

import torch
from torch import Tensor
from torch.nn.utils.clip_grad import clip_grad_norm
from easyfl.client.base import BaseClient

logger = logging.getLogger(__name__)


class Client(BaseClient):
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

        for i in range(conf.local_epoch):
            batch_loss = []
            for X, y in self.train_loader:
                X: Tensor = X.to(device).float()
                y: Tensor = y.to(device).float()

                output, M_loss = self.model(X)

                loss: Tensor = loss_fn(output, y)

                # Add the overall sparsity loss
                loss = loss - conf.lambda_sparse * M_loss

                # Perform backward pass and optimization
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
        pass
