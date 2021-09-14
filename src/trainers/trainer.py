import abc
from collections import OrderedDict
from typing import Iterable

from torch import nn as nn

from src.torch.pytorch_util import np_to_pytorch_batch


class Trainer(object, metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def train(self, data):
        pass

    def end_epoch(self, epoch):
        pass

    def get_snapshot(self):
        return {}

    def get_diagnostics(self):
        return {}



class TorchTrainer(Trainer, metaclass=abc.ABCMeta):
    def __init__(self):
        self._num_train_steps = 0

    def train(self, np_batch):
        self._num_train_steps += 1
        batch = np_to_pytorch_batch(np_batch)
        self.train_from_torch(batch)

    def get_diagnostics(self):
        return OrderedDict([
            ('num train calls', self._num_train_steps),
        ])

    def train_from_torch(self, batch):
        pass

    @property
    def networks(self) -> Iterable[nn.Module]:
        pass
