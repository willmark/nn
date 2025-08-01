import torch
import torch.nn.functional as F

# Sample dimensions
batch_size = 1
seq_len = 3
embed_dim = 4

# Simulated input embeddings
x = torch.rand(batch_size, seq_len, embed_dim)  # (1, 3, 4)

# Linear layers to compute Q, K, V
W_q = torch.nn.Linear(embed_dim, embed_dim, bias=False)
W_k = torch.nn.Linear(embed_dim, embed_dim, bias=False)
W_v = torch.nn.Linear(embed_dim, embed_dim, bias=False)

Q = W_q(x)  # (1, 3, 4)
K = W_k(x)  # (1, 3, 4)
V = W_v(x)  # (1, 3, 4)

# Scaled dot-product attention
dk = Q.size(-1)
scores = torch.matmul(Q, K.transpose(-2, -1)) / dk**0.5  # (1, 3, 3)

attention_weights = F.softmax(scores, dim=-1)  # (1, 3, 3)
output = torch.matmul(attention_weights, V)  # (1, 3, 4)

print("Attention Weights:\n", attention_weights)
print("Attention Output:\n", output)