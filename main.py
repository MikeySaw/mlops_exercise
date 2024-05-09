import click
import torch
from model import MyAwesomeModel

from data import mnist


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
    train_set, _ = mnist()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    criterion = torch.nn.NLLLoss()

    for e in range(40):
        running_loss = 0
        for images, labels in train_set:
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

        print(f"Training loss: {running_loss/len(train_set)}")

    torch.save(model.state_dict(), "model.pt")


@click.command()
@click.argument("model_checkpoint")
def evaluate(model_checkpoint="model.pt"):
    """Evaluate a trained model."""
    print("Evaluating like my life dependends on it")
    print(model_checkpoint)
    model = MyAwesomeModel()

    # TODO: Implement evaluation logic here
    model.load_state_dict(torch.load("model.pt"))
    _, test_set = mnist()
    with torch.no_grad():
        accuracy_sum = 0
        for images, labels in test_set:
            images = images.view(images.shape[0], -1)
            ps = torch.exp(model(images))
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy_sum += torch.mean(equals.type(torch.FloatTensor))

        print(f"Accuracy: {(accuracy_sum.item()*100)/len(test_set)}%")


cli.add_command(train)
cli.add_command(evaluate)


if __name__ == "__main__":
    cli()
