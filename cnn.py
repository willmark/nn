import torch
import torch.nn as nn
import torch.nn.functional as F


class ImageClassifierCNN(nn.Module):
    """
    A convolutional neural network for image classification.
    
    This model is designed for MNIST-like image classification tasks, using a simple
    CNN architecture with two convolutional layers followed by max pooling and a
    final fully connected classification layer.
    
    Architecture:
        - Conv2d(1→8, 3x3) → ReLU → MaxPool2d(2x2)
        - Conv2d(8→16, 3x3) → ReLU → MaxPool2d(2x2)
        - Flatten → Linear(400→10)
    
    Expected input shape: [batch_size, channels=1, height=28, width=28]
    Expected output shape: [batch_size, num_classes=10]
    
    Attributes:
        conv1 (nn.Conv2d): First convolutional layer (1→8 channels, 3x3 kernel)
        conv2 (nn.Conv2d): Second convolutional layer (8→16 channels, 3x3 kernel)
        pool (nn.MaxPool2d): Max pooling layer (2x2 kernel, stride 2)
        fc (nn.Linear): Final fully connected classification layer
    """
    
    def __init__(self):
        """
        Initialize the CNN image classifier.
        
        The model is designed for 28x28 grayscale images (MNIST format) and
        outputs 10 class probabilities for digit classification (0-9).
        """
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, 3)      # input: [B, 1, 28, 28] → conv1 output: [B, 8, 26, 26]
        self.conv2 = nn.Conv2d(8, 16, 3)     # input: [B, 8, 13, 13] → conv2 output: [B, 16, 11, 11]
        self.pool = nn.MaxPool2d(2, 2)       # halves H and W
        self.fc = nn.Linear(16 * 5 * 5, 10)  # final feature map after second pool: [B, 16, 5, 5]

    def forward(self, x):
        """
        Forward pass through the CNN model.
        
        Args:
            x (torch.Tensor): Input batch of images.
                Shape: [batch_size, channels=1, height=28, width=28]
                Expected to be grayscale images (1 channel) of size 28x28.
        
        Returns:
            torch.Tensor: Classification logits for each image in the batch.
                Shape: [batch_size, num_classes=10]
                The output represents unnormalized log probabilities for each digit class (0-9).
        
        Note:
            The model applies two convolutional layers with ReLU activation and max pooling,
            then flattens the features and applies a final linear layer for classification.
        """
        # conv1: [B, 1, 28, 28] → [B, 8, 26, 26]; pool: [B, 8, 13, 13]
        x = self.pool(F.relu(self.conv1(x)))
        # conv2: [B, 8, 13, 13] → [B, 16, 11, 11]; pool: [B, 16, 5, 5]
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)  # flatten: [B, 16*5*5] = [B, 400]
        return self.fc(x)


# Example usage
x = torch.randn(4, 1, 28, 28)
model = ImageClassifierCNN()
print(model(x).shape)  # [4, 10]