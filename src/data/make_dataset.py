import torch
from torch.utils.data import Dataset
import os
import pickle


# create pytorch dataset class
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
    path = "data/raw/"

    # Load test data
    test_images = torch.load(os.path.join(path + "test_images.pt"))
    test_target = torch.load(os.path.join(path + "test_target.pt"))

    # Load train data
    train_images = []
    train_target = []

    for i in range(6):  # loop over all the images in the folder
        train_images.append(torch.load(os.path.join(path + f"train_images_{i}.pt")))
        train_target.append(torch.load(os.path.join(path + f"train_target_{i}.pt")))

    # Concatenate loaded tensors
    train_images = torch.cat(train_images, dim=0)
    train_target = torch.cat(train_target, dim=0)

    # create instances of pytorch dataset class
    train_dataset = CustomDataset(train_images, train_target)
    test_dataset = CustomDataset(test_images, test_target)

    # train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    # test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    return train_dataset, test_dataset


def main():
    train, test = mnist()

    folder_path = "data/processed"

    # Make sure the folder exists, otherwise create it
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    train_path = os.path.join(folder_path, "train_set.pkl")

    with open(train_path, "wb") as f:
        pickle.dump(train, f)

    test_path = os.path.join(folder_path, "test_set.pkl")
    with open(test_path, "wb") as f:
        pickle.dump(test, f)

    print("Datasets saved to:", folder_path)


if __name__ == "__main__":
    # Get the data and process it
    main()
