# 🧠 Residual vs Plain Blocks in PyTorch

A minimal PyTorch demonstration contrasting Residual Blocks and Plain (Linear) Blocks to showcase the impact of residual connections on gradient flow during backpropagation.

## 📋 Overview

Residual connections, introduced in ResNet, are widely known to improve training in deep neural networks by mitigating vanishing gradient problems. This educational example demonstrates how residual connections:

- **Improve gradient flow** through the network
- **Maintain signal strength** across layers  
- **Alter gradients** received by earlier layers and inputs

## 🏗️ Architecture

### Block Types

The script implements two fundamental block types:

1. **ResidualBlock**: Applies a linear transformation and adds the input (skip connection)
2. **PlainBlock**: Applies the same linear transformation without any skip connection

### Network Structure

- **Depth**: 10 layers (configurable)
- **Input**: Fixed 3-dimensional tensor `[1, 2, 3]`
- **Architecture**: Sequential stack of identical blocks

## 🔬 Methodology

The script performs the following analysis:

1. **Forward Pass**: Propagates input through both networks
2. **Backward Pass**: Computes gradients with respect to:
   - Final layer activations
   - Input gradients
   - First layer weight gradients
3. **Comparison**: Side-by-side analysis of gradient magnitudes and flow

## 📊 Results

### Residual Network
```
Final Output: [[8.760121  2.0753431 2.5167747]]
Input Gradients: [[-1.8296819  4.0609355  3.6003714]]
First Layer Weight Gradients: 
[[-0.2950688 -0.5901376 -0.8852064]
 [ 3.5320826  7.064165  10.596248 ]
 [ 3.0706484  6.141297   9.211946 ]]
```

### Plain Network
```
Final Output: [[-0.49245155 -0.16423884  0.34569877]]
Input Gradients: [[ 0.00071465 -0.0004733  -0.00085846]]
First Layer Weight Gradients:
[[ 1.8755258e-03  3.7510516e-03  5.6265774e-03]
 [ 1.0151875e-03  2.0303749e-03  3.0455624e-03]
 [-8.5578213e-05 -1.7115643e-04 -2.5673464e-04]]
```

## 💡 Key Insights

- **Residual Network**: Maintains strong gradients and higher final outputs
- **Plain Network**: Shows rapid gradient attenuation, demonstrating the vanishing gradient problem
- **Magnitude Difference**: Residual gradients are orders of magnitude larger than plain network gradients

## 🚀 Quick Start

### Prerequisites
```bash
pip install -r requirements.txt
```

### Run the Demo
```bash
python compare_blocks.py
```

### Clone Repository
```bash
git clone https://github.com/willmark/residuals.git
cd residuals
```

## 📁 Project Structure

```
residuals/
├── compare_blocks.py    # Main demonstration script
├── requirements.txt     # Python dependencies
└── README.md          # This file
```

## 🔧 Dependencies

- `torch` - PyTorch deep learning framework
- `numpy` - Numerical computing library

## 📚 Further Reading

- [ResNet Paper](https://arxiv.org/abs/1512.03385) - Original residual network paper
- [PyTorch Documentation](https://pytorch.org/docs/) - PyTorch framework reference
