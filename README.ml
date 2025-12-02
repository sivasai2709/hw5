Student Name: M siva sai kumar

Student ID: 700765523

Course: CS5710- Machine Learning

Implements softmax and scaled dot-product attention using NumPy.

Calculates:

scores = Q × Kᵀ / √dₖ

attention weights = softmax(scores)

context = attention weights × V

Test code creates random Q, K, V and prints shapes of:

Attention weights: (batch, seq_len, seq_len)

Context vectors: (batch, seq_len, d_v)

Helps understand the core attention mechanism used in Transformers.
2.
This code builds a Multi-Head Self-Attention layer and a Transformer Encoder using PyTorch.

It projects input into Q, K, V, splits into heads, and computes scaled dot-product attention.

The encoder applies attention → residual → layer norm, then feed-forward → residual → layer norm.

Dropout is included for regularization.

Test code shows output shape (batch, seq_len, d_model) and attention shape (batch, heads, seq_len, seq_len)

