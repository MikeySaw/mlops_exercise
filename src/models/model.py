import torch.nn.functional as F
from torch import nn


class MyAwesomeModel(nn.Module):
    """My awesome model."""

    def __init__(self, hidden1, hidden2, hidden3, hidden4, drop_p):
        super().__init__()
        self.fc1 = nn.Linear(784, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, hidden3)
        self.fc4 = nn.Linear(hidden3, hidden4)
        self.fc5 = nn.Linear(hidden4, 10)
        self.dropout = nn.Dropout(p=drop_p)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(F.relu(self.fc2(x)))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.log_softmax(self.fc5(x), dim=-1)
        return x
