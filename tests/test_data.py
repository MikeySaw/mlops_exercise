import os.path

import pytest
import torch

from src.data.make_dataset import mnist
from tests import _PATH_DATA


@pytest.mark.skipif(not os.path.exists(_PATH_DATA), reason="Data files not found")
def test_data():
    train, test = mnist()
    assert len(train) == 30000
    assert len(test) == 5000
    for dataset in [train, test]:
        for x, y in dataset:
            assert x.shape == (1, 28, 28)
            assert y in range(10)
    train_targets = torch.unique(train.targets)
    assert (train_targets == torch.arange(0, 10)).all()
    test_targets = torch.unique(test.targets)
    assert (test_targets == torch.arange(0, 10)).all()
