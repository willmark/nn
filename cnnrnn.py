import torch
import torch.nn as nn
import torch.nn.functional as F


class SequenceImageClassifierCNNRNN(nn.Module):
    """
    A hybrid CNN-RNN model for classifying sequences of images.
    
    This model combines convolutional neural networks (CNN) and recurrent neural networks (RNN)
    to process temporal sequences of images. The CNN extracts spatial features from each image
    frame, while the RNN processes the temporal sequence of these features to make a final
    classification decision.
    
    Architecture:
        - CNN layers: Extract spatial features from each image frame
        - RNN layer: Process the temporal sequence of CNN features
        - Fully connected layer: Final classification head
    
    Expected input shape: [batch_size, sequence_length, channels, height, width]
    Expected output shape: [batch_size, num_classes]
    
    Attributes:
        conv1 (nn.Conv2d): First convolutional layer (1→8 channels, 3x3 kernel)
        conv2 (nn.Conv2d): Second convolutional layer (8→16 channels, 3x3 kernel)
        pool (nn.MaxPool2d): Max pooling layer (2x2 kernel, stride 2)
        feature_size (int): Size of flattened CNN features per frame
        rnn (nn.RNN): Recurrent neural network for sequence processing
        fc (nn.Linear): Final fully connected classification layer
    """
    
    def __init__(self):
        """
        Initialize the CNN-RNN sequence classifier.
        
        The model is designed for MNIST-like images (28x28 grayscale) and sequences
        of variable length. The CNN reduces each 28x28 image to a 400-dimensional
        feature vector, which is then processed by the RNN.
        """
        super().__init__()

        # CNN to extract features from each image frame
        self.conv1 = nn.Conv2d(1, 8, 3)       # [B*T, 1, 28, 28] → [B*T, 8, 26, 26]
        self.conv2 = nn.Conv2d(8, 16, 3)      # [B*T, 8, 13, 13] → [B*T, 16, 11, 11]
        self.pool = nn.MaxPool2d(2, 2)        # → [B*T, 16, 5, 5]

        # Feature size after CNN for each frame
        self.feature_size = 16 * 5 * 5

        # RNN to process sequence of features
        self.rnn = nn.RNN(input_size=self.feature_size, hidden_size=32, batch_first=True)

        # Final classifier
        self.fc = nn.Linear(32, 5)            # Output class size

    def forward(self, x_seq):
        """
        Forward pass through the CNN-RNN model.
        
        Args:
            x_seq (torch.Tensor): Input sequence of images.
                Shape: [batch_size, sequence_length, channels=1, height=28, width=28]
                Expected to be grayscale images (1 channel) of size 28x28.
        
        Returns:
            torch.Tensor: Classification logits for each sequence in the batch.
                Shape: [batch_size, num_classes=5]
                The output represents unnormalized log probabilities for each class.
        
        Note:
            The model processes each frame through CNN layers to extract spatial features,
            then uses an RNN to process the temporal sequence of these features.
            The final classification is based on the last hidden state of the RNN.
        """
        # x_seq shape: [B, T, C=1, H=28, W=28]
        B, T, C, H, W = x_seq.shape
        x_seq = x_seq.view(B * T, C, H, W)

        # Pass each frame through CNN
        x = self.pool(F.relu(self.conv1(x_seq)))       # → [B*T, 8, 13, 13]
        x = self.pool(F.relu(self.conv2(x)))           # → [B*T, 16, 5, 5]
        x = x.view(B, T, -1)                           # → [B, T, 400]

        # Pass the sequence of CNN features through RNN
        out, _ = self.rnn(x)                           # → [B, T, 32]
        return self.fc(out[:, -1, :])                  # Use last time step → [B, 5]


# Example usage: batch of 4 sequences, each with 6 grayscale frames of size 28x28
x = torch.randn(4, 6, 1, 28, 28)
model = SequenceImageClassifierCNNRNN()
print(model(x).shape)  # Output: [4, 5]