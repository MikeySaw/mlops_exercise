import matplotlib.pyplot as plt
import torch
from make_dataset import mnist
from mpl_toolkits.axes_grid1 import ImageGrid

def show_image_and_target(images: torch.Tensor, target: torch.Tensor) -> None:
    """Plot images and their labels in a grid."""
    row_col = int(len(images) ** 0.5)
    fig = plt.figure(figsize=(10.0, 10.0))
    grid = ImageGrid(fig, 111, nrows_ncols=(row_col, row_col), axes_pad=0.3)
    for ax, im, label in zip(grid, images, target):
        ax.imshow(im.squeeze(), cmap="gray")
        ax.set_title(f"Label: {label.item()}")
        ax.axis("off")
    plt.show()



def dataset_statistics():
    train_dataset, test_dataset = mnist()
    print(f"Train dataset: MNIST")
    print(f"Number of images: {len(train_dataset)}")
    print(f"Image shape: {train_dataset[0][0].shape}")
    print("\n")
    print(f"Test dataset: MNIST")
    print(f"Number of images: {len(test_dataset)}")
    print(f"Image shape: {test_dataset[0][0].shape}")

    show_image_and_target(train_dataset.images[:25], train_dataset.targets[:25])
    plt.savefig("reports/figures/mnist_images.png")
    plt.close()

    train_label_distribution = torch.bincount(train_dataset.targets)
    test_label_distribution = torch.bincount(test_dataset.targets)

    plt.bar(torch.arange(10), train_label_distribution)
    plt.title("Train label distribution")
    plt.xlabel("Label")
    plt.ylabel("Count")
    plt.savefig("reports/figures/train_label_distribution.png")
    plt.close()

    plt.bar(torch.arange(10), test_label_distribution)
    plt.title("Test label distribution")
    plt.xlabel("Label")
    plt.ylabel("Count")
    plt.savefig("reports/figures/test_label_distribution.png")
    plt.close()


if __name__ == "__main__":
    dataset_statistics()