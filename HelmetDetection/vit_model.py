import torch
import torch.nn as nn
from torch.nn import functional as F

EMBED_DIM = 128
PATCH_SIZE = 16
MAX_TOKENS = 1000
DEPTH = 6
NUM_CLASSES = 3
NUM_PREDICTIONS = 196

class PatchEmbedding(nn.Module):
    def __init__(self, embed_dim, patch_size):
        super().__init__()
        self.PatchEmbedding = nn.Conv2d(
            in_channels=3,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
    
    def forward(self, x):
        x = self.PatchEmbedding(x)
        B, C, H, W = x.shape
        x = x.reshape((B, C, H*W)).transpose(1, 2)
        return x
    
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.Q = nn.Linear(embed_dim, embed_dim)
        self.K = nn.Linear(embed_dim, embed_dim)
        self.V = nn.Linear(embed_dim, embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        Q = self.Q(x)
        K = self.K(x)
        V = self.V(x)
        N, H, W = Q.shape
        Q = Q.reshape(N, H, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.reshape(N, H, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.reshape(N, H, self.num_heads, self.head_dim).transpose(1, 2)
        attention_scores = Q @ K.transpose(2, 3)
        attention_scores = attention_scores / (self.head_dim ** 0.5)
        attention_probs = F.softmax(attention_scores, dim=-1)
        self.last_attn = attention_probs.detach()
        attention_output = attention_probs @ V
        attention_output = attention_output.transpose(1,2)
        attention_output = attention_output.reshape(N, H, self.embed_dim)
        attention_output = self.proj(attention_output)
        return attention_output

class FeedForward(nn.Module):
    def __init__(self, embed_dim, hidden_dim):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim)
        )

    def forward(self, x):
        x = self.layer(x)
        return x
    
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, hidden_dim):
        super().__init__()
        self.mhsa = MultiHeadSelfAttention(embed_dim, num_heads)
        self.ff = FeedForward(embed_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = x + self.mhsa(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x

class VIT(nn.Module):
    def __init__(self, embed_dim=EMBED_DIM, patch_size=PATCH_SIZE, max_tokens=MAX_TOKENS, num_classes=NUM_CLASSES):
        super().__init__()
        self.PatchEmbedding = PatchEmbedding(embed_dim, patch_size)
        self.position_embedding = nn.Parameter(torch.zeros(max_tokens, embed_dim))
        self.blocks = nn.ModuleList([TransformerBlock(embed_dim, num_heads=8, hidden_dim=embed_dim*4) for _ in range(DEPTH)])
        self.norm = nn.LayerNorm(embed_dim)
        self.classification_head = nn.Linear(embed_dim, num_classes + 1)
        self.bbox = nn.Linear(embed_dim, 4)
        self.num_predictions = NUM_PREDICTIONS
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.PatchEmbedding(x)
        x = x + self.position_embedding[:x.size(1), :]
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)

        x = x[:, :self.num_predictions, :]
        class_logits = self.classification_head(x)
        bbox_preds = self.bbox(x)
        bbox_pred = self.sigmoid(bbox_preds)

        return class_logits, bbox_pred