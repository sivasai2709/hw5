import numpy as np

def softmax(x, axis=-1):
    # Stable softmax implementation
    x_shifted = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x_shifted)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

def scaled_dot_product_attention(Q, K, V):
    """
    Computes scaled dot-product attention.

    Q: (batch, seq_len, d_k)
    K: (batch, seq_len, d_k)
    V: (batch, seq_len, d_v)
    """

    d_k = Q.shape[-1]

    # Step 1: raw scores = QK^T
    scores = np.matmul(Q, K.transpose(0, 2, 1)) / np.sqrt(d_k)

    # Step 2: softmax over last dimension → attention weights
    attn_weights = softmax(scores, axis=-1)

    # Step 3: weighted sum of values
    context = np.matmul(attn_weights, V)

    return attn_weights, context


# -------- Small Test --------
if __name__ == "__main__":
    np.random.seed(5)

    batch = 2
    seq_len = 4
    d_k = 3
    d_v = 5

    Q = np.random.rand(batch, seq_len, d_k)
    K = np.random.rand(batch, seq_len, d_k)
    V = np.random.rand(batch, seq_len, d_v)

    attn, ctx = scaled_dot_product_attention(Q, K, V)

    print("Attention weights shape:", attn.shape)
    print("Context vector shape:", ctx.shape)
