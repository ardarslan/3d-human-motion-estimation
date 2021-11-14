class Experiment(object):
    def __init__(self, cfg):
        self.cfg = cfg

    def train(self):
        raise NotImplementedError()

    def test(self):
        raise NotImplementedError()

    def visualize(self):
        raise NotImplementedError()
