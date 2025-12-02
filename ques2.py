import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------------
# Multi-Head Self Attention Layer
# ------------------------------

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must divide evenly across heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        # Linear layers for Q, K, V
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        # Final projection
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, T, _ = x.size()

        # Project inputs
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        # Split into heads: (B, num_heads, T, d_head)
        def split_heads(t):
            return t.view(B, T, self.num_heads, self.d_head).transpose(1, 2)

        Qh = split_heads(Q)
        Kh = split_heads(K)
        Vh = split_heads(V)

        # Compute scaled dot-product attention
        scores = torch.matmul(Qh, Kh.transpose(-2, -1)) / (self.d_head ** 0.5)
        weights = F.softmax(scores, dim=-1)
        out = torch.matmul(weights, Vh)

        # Merge heads back
        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)

        return self.W_o(out), weights


# ------------------------------
# Transformer Encoder Block
# ------------------------------

class TransformerEncoder(nn.Module):
    def __init__(self, d_model=128, num_heads=8, d_ff=512, dropout=0.1):
        super().__init__()

        self.attn = MultiHeadSelfAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )

        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        # Self-attention + residual
        attn_out, attn_weights = self.attn(x)
        x = self.norm1(x + self.drop(attn_out))

        # Feed-forward + residual
        ff_out = self.ff(x)
        x = self.norm2(x + self.drop(ff_out))

        return x, attn_weights


# ------------------------------
# Test with Sample Input
# ------------------------------
if __name__ == "__main__":
    batch = 32
    seq_len = 10
    d_model = 128

    x = torch.randn(batch, seq_len, d_model)

    encoder = TransformerEncoder(d_model=128, num_heads=8, d_ff=256)
    out, attn = encoder(x)

    print("Output shape:", out.shape)
    print("Attention shape:", attn.shape)
