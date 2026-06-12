import os
import matplotlib.pyplot as plt
from src.config import CHECKPOINTS_FOLDER

def plot_losses(train_losses, val_losses, save_name, save_path=CHECKPOINTS_FOLDER):
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, label="Train")
    plt.plot(epochs, val_losses, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Progress")
    plt.legend()
    plt.grid(True)
    if save_name:
        plt.savefig(os.path.join(save_path, save_name))
    plt.show()