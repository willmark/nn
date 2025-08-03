import torch
import torch.nn as nn

torch.manual_seed(0)


class ResidualBlock(nn.Module):
    """
    A residual block that implements skip connections.
    
    This block adds the input to the output of a linear transformation,
    creating a residual connection that helps with gradient flow in deep networks.
    The residual connection allows the network to learn residual functions
    rather than complete transformations.
    
    Architecture:
        - Linear transformation: input → linear_layer(input)
        - Residual connection: input + linear_layer(input)
    
    Expected input shape: [batch_size, size]
    Expected output shape: [batch_size, size]
    
    Attributes:
        lin (nn.Linear): Linear transformation layer
    """
    
    def __init__(self, size):
        """
        Initialize the residual block.
        
        Args:
            size (int): Input and output dimension for the linear layer.
        """
        super().__init__()
        self.lin = nn.Linear(size, size)
    
    def forward(self, x):
        """
        Forward pass through the residual block.
        
        Args:
            x (torch.Tensor): Input tensor.
                Shape: [batch_size, size]
        
        Returns:
            torch.Tensor: Output with residual connection.
                Shape: [batch_size, size]
                Computed as: x + linear_layer(x)
        """
        return x + self.lin(x)


class PlainBlock(nn.Module):
    """
    A plain block without skip connections.
    
    This block applies only a linear transformation without any residual connections.
    It serves as a comparison to demonstrate the benefits of residual connections
    in deep networks.
    
    Architecture:
        - Linear transformation: input → linear_layer(input)
    
    Expected input shape: [batch_size, size]
    Expected output shape: [batch_size, size]
    
    Attributes:
        lin (nn.Linear): Linear transformation layer
    """
    
    def __init__(self, size):
        """
        Initialize the plain block.
        
        Args:
            size (int): Input and output dimension for the linear layer.
        """
        super().__init__()
        self.lin = nn.Linear(size, size)
    
    def forward(self, x):
        """
        Forward pass through the plain block.
        
        Args:
            x (torch.Tensor): Input tensor.
                Shape: [batch_size, size]
        
        Returns:
            torch.Tensor: Output after linear transformation.
                Shape: [batch_size, size]
                Computed as: linear_layer(x)
        """
        return self.lin(x)


def make_net(block, size=3, depth=10):
    """
    Create a sequential network using the specified block type.
    
    This function creates a deep network by stacking multiple instances of the
    specified block type. It's used to compare the behavior of residual vs plain
    blocks in deep architectures.
    
    Args:
        block (type): Block class to use (ResidualBlock or PlainBlock).
        size (int): Input/output dimension for each block.
            Default: 3
        depth (int): Number of blocks to stack in the network.
            Default: 10
    
    Returns:
        nn.Sequential: A sequential network containing 'depth' blocks.
    
    Example:
        >>> res_net = make_net(ResidualBlock, size=3, depth=10)
        >>> plain_net = make_net(PlainBlock, size=3, depth=10)
    """
    return nn.Sequential(*[block(size) for _ in range(depth)])


def run(model, x_input):
    """
    Run a forward pass through the model and compute gradients.
    
    This function performs a forward pass through the model, tracks activations
    at each layer, computes gradients, and returns the results for analysis.
    It's used to compare the behavior of residual vs plain networks.
    
    Args:
        model (nn.Module): The neural network model to run.
        x_input (torch.Tensor): Input tensor with requires_grad=True.
            Shape: [batch_size, size]
    
    Returns:
        tuple: A tuple containing:
            - acts (list): List of activations from each layer
            - grad_x (torch.Tensor): Gradient with respect to input
            - grad_w (torch.Tensor): Gradient with respect to first layer weights
    
    Note:
        The function computes the sum of the final output and performs backpropagation
        to analyze gradient flow through the network.
    """
    acts = []
    x = x_input
    for layer in model:
        x = layer(x)
        acts.append(x)
    y = x.sum()
    y.backward()
    return acts, x_input.grad, model[0].lin.weight.grad


# Create networks
res_net = make_net(ResidualBlock)
plain_net = make_net(PlainBlock)

# Shared input
x_res = torch.tensor([[1., 2., 3.]], requires_grad=True)
x_plain = x_res.clone().detach().requires_grad_()

# Run both
acts_r, grad_x_r, grad_w_r = run(res_net, x_res)
acts_p, grad_x_p, grad_w_p = run(plain_net, x_plain)

# Results
print("Residual - Final Output:", acts_r[-1].detach().numpy())
print("Residual - Grad Input:", grad_x_r.detach().numpy())
print("Residual - Grad W[0]:", grad_w_r.detach().numpy())

print("\nPlain - Final Output:", acts_p[-1].detach().numpy())
print("Plain - Grad Input:", grad_x_p.detach().numpy())
print("Plain - Grad W[0]:", grad_w_p.detach().numpy())
