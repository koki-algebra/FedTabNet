from easyfl.server.base import BaseServer


class Server(BaseServer):
    def __init__(
        self, conf, test_data=None, val_data=None, is_remote=False, local_port=22999
    ):
        super().__init__(conf, test_data, val_data, is_remote, local_port)
