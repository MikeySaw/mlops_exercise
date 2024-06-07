import pytest
import torch

from src.models.model import MyAwesomeModel


@pytest.mark.parametrize("batch_size", [32, 64])
def test_model(batch_size: int) -> None:
    model = MyAwesomeModel(500, 250, 125, 60, 0.2)
    x = torch.randn(batch_size, 1, 28, 28)
    x = x.view(x.shape[0], -1)
    y = model(x)
    assert y.shape == (batch_size, 10)
