import click
import torch
import pickle
import os
from models.model import MyAwesomeModel
from torch.utils.data import DataLoader
from data.make_dataset import CustomDataset
import matplotlib.pyplot as plt


def create_plot(losses):
    epochs = range(1, len(losses) + 1)

    plt.plot(epochs, losses, label="Training Losses")
    plt.xlabel("Epochs")
    plt.ylabel("Losses")
    plt.title("Training Losses Over Epochs")

    folder_path = "reports/figures"
    file_path = os.path.join(folder_path, "training_losses_plot.png")
    plt.savefig(file_path)

    print("Plot saved to:", file_path)


@click.group()
def cli():
    """Command line interface."""
    pass


@click.command()
@click.option("--lr", default=1e-3, help="learning rate to use for training")
def train(lr):
    """Train a model on MNIST."""
    print("Training day and night")
    print(lr)

    # TODO: Implement training loop here

    model = MyAwesomeModel()
    with open("data/processed/train_set.pkl", "rb") as f:
        train_set = pickle.load(f)

    train_dataloader = DataLoader(train_set, batch_size=64, shuffle=True)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    criterion = torch.nn.NLLLoss()
    training_losses = []

    for e in range(40):
        running_loss = 0
        for images, labels in train_dataloader:
            # Flatten MNIST images into a 784 long vector
            images = images.view(images.shape[0], -1)
            # print(images.shape, labels.shape)

            # TODO: Training pass
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        training_losses.append(running_loss / len(train_dataloader))

        print(f"Training loss: {running_loss/len(train_dataloader)}")

    torch.save(model.state_dict(), "models/model.pt")
    print("Model has been saved to models/model.pt")
    create_plot(training_losses)


cli.add_command(train)


if __name__ == "__main__":
    cli()
