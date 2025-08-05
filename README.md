# 🧠 Neural Network Examples in PyTorch

A collection of minimal PyTorch demonstrations showcasing fundamental neural network concepts and architectures. **These are toy examples designed for educational purposes - useful for lectures, presentations, tutorials, and learning deep learning concepts.**

## 📋 Overview

This repository contains practical examples that demonstrate key concepts in deep learning. Each example is intentionally simplified to focus on core concepts rather than production-ready implementations.

## 🎯 Purpose

These examples are specifically designed for:
- **Lectures and Presentations**: Clear, minimal code that can be easily explained
- **Educational Tutorials**: Self-contained examples with comprehensive documentation
- **Concept Learning**: Focus on understanding rather than optimization
- **Teaching Demonstrations**: Visual and interactive learning materials

## 🏗️ Examples

### 1. Convolutional Neural Network (`cnn.py`)

Simple CNN for MNIST digit classification demonstrating convolutional layers, pooling, and classification.

**Key Concepts:**
- **Convolutional Layers**: Feature extraction from images
- **Max Pooling**: Dimensionality reduction
- **Fully Connected Classification**: Final prediction layer

**Architecture:**
- Input: `(batch_size, 1, 28, 28)` - MNIST grayscale images
- Output: `(batch_size, 10)` - Digit classification (0-9)

### 2. Recurrent Neural Network (`rnn.py`)

Basic RNN implementation from scratch for sequence classification.

**Key Concepts:**
- **Hidden State Management**: Manual implementation of RNN cell
- **Sequence Processing**: Step-by-step processing of sequential data
- **Parameter Management**: Direct control over weights and biases

**Architecture:**
- Input: `(batch_size, sequence_length, input_size)`
- Output: `(batch_size, output_size)`

### 3. CNN-RNN Hybrid (`cnnrnn.py`)

Combines CNN and RNN for processing sequences of images.

**Key Concepts:**
- **Feature Extraction**: CNN processes individual frames
- **Temporal Modeling**: RNN processes sequence of features
- **Multi-modal Architecture**: Combining spatial and temporal information

**Architecture:**
- Input: `(batch_size, sequence_length, 1, 28, 28)` - Sequence of images
- Output: `(batch_size, 5)` - Sequence classification

### 4. Residual vs Plain Blocks (`compare_blocks.py`)

Demonstrates the impact of residual connections on gradient flow during backpropagation.

**Key Concepts:**
- **ResidualBlock**: Applies linear transformation and adds input (skip connection)
- **PlainBlock**: Applies same linear transformation without skip connection
- **Gradient Flow**: Comparison of gradient magnitudes and vanishing gradient problem

**Results:**
- Residual networks maintain stronger gradients and higher outputs
- Plain networks show rapid gradient attenuation through depth

### 5. Attention Mechanism (`attention.py`)

Basic implementation of scaled dot-product attention mechanism.

**Key Concepts:**
- **Query, Key, Value**: Linear transformations of input embeddings
- **Attention Weights**: Softmax of scaled dot-product scores
- **Output**: Weighted combination of values using attention weights

**Architecture:**
- Input: `(batch_size, seq_len, embed_dim)`
- Output: `(batch_size, seq_len, embed_dim)`

### 6. Autoencoder (`autoencoder.py`)

Simple autoencoder for dimensionality reduction and feature learning through unsupervised reconstruction.

**Key Concepts:**
- **Unsupervised Learning**: Learning without labels through reconstruction
- **Dimensionality Reduction**: Compressing high-dimensional data into latent space
- **Feature Learning**: Discovering meaningful compressed representations
- **Bottleneck Architecture**: Forcing information compression through narrow latent space
- **Reconstruction Loss**: Measuring how well original data is preserved

**Architecture:**
- **Encoder**: `input_dim → 128 → latent_dim` (compression)
- **Decoder**: `latent_dim → 128 → input_dim` (reconstruction)
- **Loss**: Mean Squared Error between input and reconstruction

**Applications:**
- Data compression and denoising
- Feature extraction for downstream tasks
- Anomaly detection through reconstruction error
- Dimensionality reduction for visualization

### 7. PyTorch Lightning Training (`lightning.py`)

Complete training pipeline using PyTorch Lightning framework.

**Key Concepts:**
- **DataModule**: Organized data loading and preprocessing
- **LightningModule**: Structured model with training/validation/test steps
- **Training Pipeline**: End-to-end training with automatic logging

**Features:**
- MNIST dataset handling
- Automatic train/val/test splits
- Metric tracking and logging
- Production-ready training structure

## 🔬 Methodology

Each example follows a consistent approach:

1. **Clear Implementation**: Minimal, readable code focused on concepts
2. **Educational Focus**: Comprehensive documentation and inline comments
3. **Practical Output**: Visual results and comparisons
4. **Self-Contained**: No external dependencies beyond PyTorch
5. **Toy Examples**: Simplified for learning, not production use

## 📊 Example Outputs

### CNN Classification
```python
x = torch.randn(4, 1, 28, 28)
model = ImageClassifierCNN()
print(model(x).shape)  # [4, 10]
```

### RNN Sequence Processing
```python
x = torch.randn(4, 7, 6)
model = SequenceClassifierRNN()
print(model(x).shape)  # [4, 3]
```

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

### Autoencoder Reconstruction
```python
x = torch.randn(4, 784)  # Flattened MNIST images
model = SimpleAutoencoder(input_dim=784, latent_dim=32)
reconstructed = model(x)
print(reconstructed.shape)  # [4, 784]
print("Reconstruction loss:", loss.item())
```

## 💡 Key Insights

- **CNN**: Effective for spatial feature extraction from images
- **RNN**: Suitable for sequential data processing
- **Hybrid Architectures**: Combine different neural network types
- **Residual Connections**: Essential for training deep networks effectively
- **Attention Mechanisms**: Foundation for modern transformer architectures
- **Autoencoders**: Unsupervised learning through reconstruction and dimensionality reduction
- **Training Frameworks**: PyTorch Lightning provides structured training

## 🚀 Quick Start

### Prerequisites
```bash
pip install -r requirements.txt
```

### Run Examples
```bash
# CNN for image classification
python cnn.py

# RNN for sequence classification
python rnn.py

# CNN-RNN hybrid for sequence of images
python cnnrnn.py

# Residual vs Plain blocks comparison
python compare_blocks.py

# Attention mechanism demonstration
python attention.py

# Autoencoder for dimensionality reduction
python autoencoder.py

# PyTorch Lightning training pipeline
python lightning.py
```

### Clone Repository
```bash
git clone https://github.com/yourusername/nn.git
cd nn
```

## 📁 Project Structure

```
nn/
├── cnn.py              # Convolutional Neural Network for MNIST
├── rnn.py              # Recurrent Neural Network from scratch
├── cnnrnn.py           # CNN-RNN hybrid for sequence classification
├── compare_blocks.py   # Residual vs Plain blocks comparison
├── attention.py        # Basic attention mechanism
├── autoencoder.py      # Autoencoder for dimensionality reduction
├── lightning.py        # PyTorch Lightning training pipeline
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

## 🔧 Dependencies

- `torch` - PyTorch deep learning framework
- `torchvision` - Computer vision utilities (for MNIST dataset)
- `pytorch-lightning` - Training framework
- `torchmetrics` - Metrics for evaluation
- `numpy` - Numerical computing library

## 🧪 Testing

Run tests to verify functionality:
```bash
# Install pytest if not already installed
pip install pytest

# Run all tests
pytest test_*.py

# Run specific test file
pytest test_cnn.py
```

## 📚 Further Reading

- [ResNet Paper](https://arxiv.org/abs/1512.03385) - Original residual network paper
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Transformer paper
- [PyTorch Documentation](https://pytorch.org/docs/) - PyTorch framework reference
- [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/) - Training framework documentation

## ⚠️ Important Note

**These are educational toy examples, not production-ready implementations.** They are designed to:
- Demonstrate core concepts clearly
- Be easily understandable for teaching
- Focus on learning rather than optimization
- Provide a foundation for more complex implementations

For production use, consider:
- More sophisticated architectures
- Proper hyperparameter tuning
- Robust error handling
- Performance optimization
- Comprehensive testing