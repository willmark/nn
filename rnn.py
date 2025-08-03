import torch
import torch.nn as nn
import torch.nn.functional as F


class SequenceClassifierRNN(nn.Module):
    """
    A recurrent neural network for sequence classification.
    
    This model implements a simple RNN from scratch using PyTorch parameters.
    It processes sequential data and outputs classification scores for the entire sequence.
    The model uses tanh activation for the hidden state updates and processes the
    sequence step by step, using the final hidden state for classification.
    
    Architecture:
        - RNN cell with input-to-hidden and hidden-to-hidden connections
        - Tanh activation for hidden state updates
        - Final fully connected layer for classification
    
    Expected input shape: [batch_size, sequence_length, input_size]
    Expected output shape: [batch_size, output_size]
    
    Attributes:
        W_ih (nn.Parameter): Input-to-hidden weight matrix
        W_hh (nn.Parameter): Hidden-to-hidden weight matrix
        b_h (nn.Parameter): Hidden layer bias
        fc (nn.Linear): Final classification layer
    """
    
    def __init__(self, input_size=6, hidden_size=12, output_size=3):
        """
        Initialize the RNN sequence classifier.
        
        Args:
            input_size (int): Dimension of input features at each time step.
                Default: 6
            hidden_size (int): Dimension of the hidden state.
                Default: 12
            output_size (int): Number of output classes for classification.
                Default: 3
        """
        super().__init__()

        # Input-to-hidden weights: transforms input vector to hidden dimension
        self.W_ih = nn.Parameter(torch.randn(input_size, hidden_size))

        # Hidden-to-hidden weights: used to update the hidden state from previous step
        self.W_hh = nn.Parameter(torch.randn(hidden_size, hidden_size))

        # Bias for the hidden layer
        self.b_h = nn.Parameter(torch.zeros(hidden_size))

        # Final fully connected layer: maps last hidden state to output class scores
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """
        Forward pass through the RNN model.
        
        Args:
            x (torch.Tensor): Input sequence of features.
                Shape: [batch_size, sequence_length, input_size]
                Each time step contains a vector of input_size dimensions.
        
        Returns:
            torch.Tensor: Classification logits for each sequence in the batch.
                Shape: [batch_size, output_size]
                The output represents unnormalized log probabilities for each class.
        
        Note:
            The model processes the sequence step by step, updating the hidden state
            at each time step using tanh activation. The final hidden state is used
            for classification through a fully connected layer.
        """
        h = torch.zeros(x.size(0), self.W_hh.size(0))
        for t in range(x.size(1)):
            # This line updates the memory (hidden state) at each step
            h = torch.tanh(x[:, t, :] @ self.W_ih + h @ self.W_hh + self.b_h)
        return self.fc(h)


# Example usage
x = torch.randn(4, 7, 6)
model = SequenceClassifierRNN()
print(model(x).shape)  # [4, 3]