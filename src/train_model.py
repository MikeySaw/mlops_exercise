import logging
import os
import pickle

import hydra
import matplotlib.pyplot as plt
import torch
import wandb
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from data.make_dataset import CustomDataset  # noqa
from models.model import MyAwesomeModel

log = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_plot(losses, plot_name):
    epochs = range(1, len(losses) + 1)

    plt.plot(epochs, losses, label="Training Losses")
    plt.xlabel("Epochs")
    plt.ylabel("Losses")
    plt.title("Training Losses Over Epochs")

    folder_path = "reports/figures"
    file_path = os.path.join(folder_path, plot_name)
    plt.savefig(file_path)

    print("Plot saved to:", file_path)


@hydra.main(config_path="config", config_name="default_config.yaml")
def train(config):
    """Train a model on MNIST."""
    print(f"configuration: \n {OmegaConf.to_yaml(config)}")
    print(os.getcwd())

    # get hyperparameters from hydras config files
    hparams = config.experiment
    model_params = config.model_conf
    torch.manual_seed(hparams["seed"])

    # setup wandb config with hydra
    wandb_config = OmegaConf.to_container(config, resolve=True, throw_on_missing=True)

    run = wandb.init(project="MLOPs wandb practice", config=wandb_config)

    # Implement training loop here
    os.chdir("../../..")
    print(os.getcwd())
    model = MyAwesomeModel(
        model_params["hidden1"],
        model_params["hidden1"],
        model_params["hidden1"],
        model_params["hidden1"],
        model_params["drop_p"],
    ).to(device)

    with open("data/processed/train_set.pkl", "rb") as f:
        train_set = pickle.load(f)

    train_dataloader = DataLoader(train_set, batch_size=hparams["batch_size"], shuffle=True)
    optimizer = torch.optim.SGD(model.parameters(), lr=hparams["lr"])
    criterion = torch.nn.NLLLoss()
    training_losses = []

    log.info("Start training MLP...")
    for e in range(hparams["n_epochs"]):
        running_loss = 0
        for images, labels in train_dataloader:
            # Flatten MNIST images into a 784 long vector
            images = images.view(images.shape[0], -1).to(device)
            # print(images.shape, labels.shape)

            # TODO: Training pass
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels.to(device))
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        wandb.log({"Loss": running_loss / len(train_dataloader)})
        training_losses.append(running_loss / len(train_dataloader))

        log.info(f"Training loss: {running_loss/len(train_dataloader)}")

    torch.save(model.state_dict(), f"models/model_{hparams["plot_name"]}_{model_params["plot_name"]}.pt")
    log.info("Model has been saved to models/model.pt")
    create_plot(training_losses, f"{hparams["plot_name"]}_{model_params["plot_name"]}.png")


if __name__ == "__main__":
    train()
