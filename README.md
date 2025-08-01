# 🧠 Neural Network Examples in PyTorch

A collection of minimal PyTorch demonstrations showcasing fundamental neural network concepts and architectures. Each example is designed to be educational, self-contained, and easy to understand.

## 📋 Overview

This repository contains practical examples that demonstrate key concepts in deep learning:

- **Residual Connections**: How skip connections improve gradient flow
- **Attention Mechanisms**: Basic scaled dot-product attention implementation

## 🏗️ Examples

### 1. Residual vs Plain Blocks (`compare_blocks.py`)

Demonstrates the impact of residual connections on gradient flow during backpropagation.

**Key Concepts:**
- **ResidualBlock**: Applies linear transformation and adds input (skip connection)
- **PlainBlock**: Applies same linear transformation without skip connection
- **Gradient Flow**: Comparison of gradient magnitudes and vanishing gradient problem

**Results:**
- Residual networks maintain stronger gradients and higher outputs
- Plain networks show rapid gradient attenuation through depth

### 2. Attention Mechanism (`attention.py`)

Basic implementation of scaled dot-product attention mechanism.

**Key Concepts:**
- **Query, Key, Value**: Linear transformations of input embeddings
- **Attention Weights**: Softmax of scaled dot-product scores
- **Output**: Weighted combination of values using attention weights

**Architecture:**
- Input: `(batch_size, seq_len, embed_dim)`
- Output: `(batch_size, seq_len, embed_dim)`

## 🔬 Methodology

Each example follows a consistent approach:

1. **Clear Implementation**: Minimal, readable code
2. **Educational Focus**: Comments explaining key concepts
3. **Practical Output**: Visual results and comparisons
4. **Self-Contained**: No external dependencies beyond PyTorch

## 📊 Example Outputs

### Residual vs Plain Blocks
```
Residual Network:
Final Output: [[8.760121  2.0753431 2.5167747]]
Input Gradients: [[-1.8296819  4.0609355  3.6003714]]

Plain Network:
Final Output: [[-0.49245155 -0.16423884  0.34569877]]
Input Gradients: [[ 0.00071465 -0.0004733  -0.00085846]]
```

### Attention Mechanism
```
Attention Weights: [[0.3333, 0.3333, 0.3333], ...]
Attention Output: [[tensor values...]]
```

## 💡 Key Insights

- **Residual Connections**: Essential for training deep networks effectively
- **Attention Mechanisms**: Foundation for modern transformer architectures
- **Gradient Flow**: Critical for understanding network training dynamics

## 🚀 Quick Start

### Prerequisites
```bash
pip install -r requirements.txt
```

### Run Examples
```bash
# Residual vs Plain Blocks
python compare_blocks.py

# Attention Mechanism
python attention.py
```

### Clone Repository
```bash
git clone https://github.com/willmark/residuals.git
cd residuals
```

## 📁 Project Structure

```
residuals/
├── compare_blocks.py    # Residual vs Plain blocks comparison
├── attention.py         # Basic attention mechanism
├── requirements.txt     # Python dependencies
└── README.md          # This file
```

## 🔧 Dependencies

- `torch` - PyTorch deep learning framework
- `numpy` - Numerical computing library



## 📚 Further Reading

- [ResNet Paper](https://arxiv.org/abs/1512.03385) - Original residual network paper
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Transformer paper
- [PyTorch Documentation](https://pytorch.org/docs/) - PyTorch framework reference

## 🤝 Contributing

Feel free to contribute additional examples or improvements to existing ones. Each example should be:
- Self-contained and runnable
- Well-documented with clear comments
- Educational and focused on key concepts
