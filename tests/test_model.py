import pytest 
import torch
from src.models.model import MyAwesomeModel

def test_model():
    model = MyAwesomeModel(500, 250, 125, 60, 0.2)
    x = torch.randn(32, 1, 28, 28)
    x = x.view(x.shape[0], -1)
    y = model(x)
    assert y.shape == (32, 10)