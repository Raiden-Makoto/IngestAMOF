import torch #type: ignore
import torch.nn as nn #type: ignore
import math

class SinusoidalTimeEmbeddings(nn.Module):
    """
    Standard Transformer-style time embeddings.
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / max(half_dim - 1, 1)  # Avoid division by zero
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings