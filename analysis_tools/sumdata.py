from collections import namedtuple
import numpy as np


class SumData(namedtuple('SumDataTuple', ['cnt', 'loss', 'vector', 'embedding'])):
    """data structure to aggregate count, loss, vector, embedding
    """
    @property
    def mean_vector(self):
        return self.vector / np.expand_dims(self.cnt, -1)

    @property
    def mean_loss(self):
        return self.loss / self.cnt

    @property
    def ppl(self):
        return min(np.exp(self.mean_loss), 99999.99)

    def __add__(self, b):
        return SumData(
            cnt = self.cnt + b.cnt,
            loss = self.loss + b.loss,
            vector = self.vector + b.vector,
            embedding = self.embedding,
        )

    def __sub__(self, b):
        return SumData(
            cnt = self.cnt - b.cnt,
            loss = self.loss - b.loss,
            vector = self.vector - b.vector,
            embedding = self.embedding,
        )

    def to_numpy(self):
        return SumData(
            cnt = self.cnt,
            loss = self.loss,
            vector = self.vector.cpu().numpy(),
            embedding = self.embedding.cpu().numpy() if self.embedding else self.embedding
        )


def zero_sum_data(hidden_dim, shape=()):
    return SumData(
        cnt = np.zeros(shape, dtype=int),
        loss = np.zeros(shape),
        vector = np.zeros(shape + (hidden_dim,)),
        embedding = None,
    )


def zero_sum_data_like(sum_data):
    return zero_sum_data(sum_data.vector.shape[-1], shape=sum_data.cnt.shape)
