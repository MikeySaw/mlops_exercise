import torch
import os
from torch.utils.data import Dataset, DataLoader


class CustomDataset(Dataset):
    def __init__(self, images, targets):
        self.images = images
        self.targets = targets

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        target = self.targets[idx]
        return image, target


def mnist():
    path = "/home/mikey/repos/mlops_exercise/data/raw/"

    # Load test data
    test_images = torch.load(os.path.join(path + "test_images.pt"))
    test_target = torch.load(os.path.join(path + "test_target.pt"))

    # Load train data
    train_images = []
    train_target = []
    for i in range(6):  # Assuming you have files train_images_0.pt to train_images_5.pt
        train_images.append(torch.load(os.path.join(path + f"train_images_{i}.pt")))
        train_target.append(torch.load(os.path.join(path + f"train_target_{i}.pt")))

    # Concatenate train data
    train_images = torch.cat(train_images, dim=0)
    train_target = torch.cat(train_target, dim=0)

    train_dataset = CustomDataset(train_images, train_target)
    test_dataset = CustomDataset(test_images, test_target)

    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    return train_dataloader, test_dataloader
