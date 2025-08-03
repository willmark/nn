import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
from torchmetrics.classification import MulticlassAccuracy


class MNISTDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for MNIST dataset.
    
    This class handles the loading, preprocessing, and splitting of the MNIST dataset
    for training, validation, and testing. It automatically downloads the dataset
    and creates appropriate DataLoaders for each stage.
    
    The dataset is split into:
        - Training: 55,000 samples
        - Validation: 5,000 samples (from training set)
        - Testing: 10,000 samples (official test set)
    
    Attributes:
        data_dir (str): Directory to store/load the MNIST dataset
        batch_size (int): Batch size for all DataLoaders
        transform (transforms.ToTensor): Transformation to convert images to tensors
        mnist_train (Dataset): Training dataset
        mnist_val (Dataset): Validation dataset
        mnist_test (Dataset): Test dataset
    """
    
    def __init__(self, data_dir='./', batch_size=64):
        """
        Initialize the MNIST DataModule.
        
        Args:
            data_dir (str): Directory to store/load the MNIST dataset.
                Default: './'
            batch_size (int): Batch size for all DataLoaders.
                Default: 64
        """
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.transform = transforms.ToTensor()

    def prepare_data(self):
        """
        Download MNIST dataset if not already present.
        
        This method is called once per node and is used to download or prepare
        the dataset. It ensures the dataset is available before setup() is called.
        """
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        """
        Set up the dataset splits for training, validation, and testing.
        
        This method is called on every process in DDP. It creates the train/val/test
        splits from the full MNIST dataset.
        
        Args:
            stage (str, optional): Current stage ('fit', 'validate', 'test', 'predict').
                If None, sets up all stages.
        """
        mnist_full = MNIST(self.data_dir, train=True, transform=self.transform)
        self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])
        self.mnist_test = MNIST(self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        """
        Create DataLoader for training data.
        
        Returns:
            DataLoader: DataLoader for training dataset with specified batch size.
        """
        return DataLoader(self.mnist_train, batch_size=self.batch_size)

    def val_dataloader(self):
        """
        Create DataLoader for validation data.
        
        Returns:
            DataLoader: DataLoader for validation dataset with specified batch size.
        """
        return DataLoader(self.mnist_val, batch_size=self.batch_size)

    def test_dataloader(self):
        """
        Create DataLoader for test data.
        
        Returns:
            DataLoader: DataLoader for test dataset with specified batch size.
        """
        return DataLoader(self.mnist_test, batch_size=self.batch_size)


class LitMNIST(pl.LightningModule):
    """
    PyTorch Lightning module for MNIST digit classification.
    
    This module implements a simple feedforward neural network for classifying
    MNIST handwritten digits. It uses PyTorch Lightning's training framework
    with automatic logging and metric tracking.
    
    Architecture:
        - Flatten: [B, 1, 28, 28] → [B, 784]
        - Linear: [B, 784] → [B, 128] → ReLU
        - Linear: [B, 128] → [B, 64] → ReLU
        - Linear: [B, 64] → [B, 10]
    
    Expected input shape: [batch_size, 1, 28, 28]
    Expected output shape: [batch_size, 10]
    
    Attributes:
        net (nn.Sequential): The neural network architecture
        accuracy (MulticlassAccuracy): Accuracy metric for tracking performance
    """
    
    def __init__(self):
        """
        Initialize the MNIST classification model.
        
        The model uses a simple feedforward architecture with three linear layers
        and ReLU activations. It's designed for 28x28 grayscale images and outputs
        10 class probabilities for digit classification (0-9).
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )
        self.accuracy = MulticlassAccuracy(num_classes=10)
    
    def forward(self, x):
        """
        Forward pass through the neural network.
        
        Args:
            x (torch.Tensor): Input batch of images.
                Shape: [batch_size, 1, 28, 28]
                Expected to be grayscale images (1 channel) of size 28x28.
        
        Returns:
            torch.Tensor: Classification logits for each image.
                Shape: [batch_size, 10]
                The output represents unnormalized log probabilities for each digit class (0-9).
        """
        return self.net(x)

    def training_step(self, batch, batch_idx):
        """
        Training step for the model.
        
        Args:
            batch (tuple): A tuple containing (images, labels).
                - images: [batch_size, 1, 28, 28]
                - labels: [batch_size]
            batch_idx (int): Index of the current batch.
        
        Returns:
            torch.Tensor: Training loss for this batch.
        
        Note:
            The loss is automatically logged to the training metrics.
        """
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Validation step for the model.
        
        Args:
            batch (tuple): A tuple containing (images, labels).
                - images: [batch_size, 1, 28, 28]
                - labels: [batch_size]
            batch_idx (int): Index of the current batch.
        
        Note:
            The accuracy is automatically logged to the validation metrics
            and displayed in the progress bar.
        """
        x, y = batch
        logits = self(x)
        acc = self.accuracy(logits.softmax(dim=-1), y)
        self.log("val_acc", acc, prog_bar=True)

    def test_step(self, batch, batch_idx):
        """
        Test step for the model.
        
        Args:
            batch (tuple): A tuple containing (images, labels).
                - images: [batch_size, 1, 28, 28]
                - labels: [batch_size]
            batch_idx (int): Index of the current batch.
        
        Note:
            The accuracy is automatically logged to the test metrics.
        """
        x, y = batch
        logits = self(x)
        acc = self.accuracy(logits.softmax(dim=-1), y)
        self.log("test_acc", acc)

    def configure_optimizers(self):
        """
        Configure the optimizer for training.
        
        Returns:
            torch.optim.Adam: Adam optimizer with learning rate 1e-3.
        """
        return torch.optim.Adam(self.parameters(), lr=1e-3)


# Training execution
if __name__ == "__main__":
    pl.seed_everything(42)

    dm = MNISTDataModule()
    model = LitMNIST()

    trainer = pl.Trainer(max_epochs=5, accelerator="auto")

    trainer.fit(model, datamodule=dm)

    print(f"Device: {model.device}")   

    trainer.test(model, datamodule=dm)