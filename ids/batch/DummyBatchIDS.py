from ids.ids import BatchIDS

"""
    DummyBatchIDS for testing and demonstration purpose only
"""


class DummyBatchIDS(BatchIDS):
    _name = "DummyBatchIDS"
    _description = ""
    _requires = ["train.ipal", "live.ipal", "train.state", "live.state", "live.batch"]
    _batch_default_settings = {}

    def __init__(self, name=None):
        super().__init__(name=name)
        self._add_default_settings(self._batch_default_settings)

    def train(self, ipal=None, state=None):
        pass

    def new_state_msg(self, msg):
        raise NotImplementedError

    def new_ipal_msg(self, msg):
        raise NotImplementedError

    def new_batch(self, batch):
        return [True for _ in range(len(batch))], [1.0 for _ in range(len(batch))]

    def save_trained_model(self):
        return True

    def load_trained_model(self):
        return True

    def visualize_model(self):
        pass
