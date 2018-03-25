from tensorflow.contrib.rnn import BasicLSTMCell


class LSTMShallowFusionCell(BasicLSTMCell):
    def __init__(self, num_utils, build_storage, storage, **kwargs):
        super().__init__(num_utils, **kwargs)
