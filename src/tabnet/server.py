from easyfl.server.base import BaseServer

from .strategies import federated_averaging


class Server(BaseServer):
    def __init__(
        self, conf, test_data=None, val_data=None, is_remote=False, local_port=22999
    ):
        super().__init__(conf, test_data, val_data, is_remote, local_port)

    def aggregate(self, models, weights):
        model = federated_averaging(models, weights)

        return model
