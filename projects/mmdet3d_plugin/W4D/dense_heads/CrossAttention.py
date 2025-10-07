import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.activation import MultiheadAttention

class CrossAttention(nn.Module):
    def __init__(self, hidden_channel=256, num_heads=8, dim_feedforward=2048, dropout=0.0):
        super(CrossAttention, self).__init__()
        self.multihead_attn = MultiheadAttention(hidden_channel, num_heads, dropout=dropout)
        self.linear1 = nn.Linear(hidden_channel, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, hidden_channel)

        self.norm1 = nn.LayerNorm(hidden_channel)
        self.norm2 = nn.LayerNorm(hidden_channel)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, query, key):
        """
        query: B C Pq
        key: B C Pk
        """
        query = query.permute(2, 0, 1)
        key = key.permute(2, 0, 1)

        # Skip self-attention
        query1, weights = self.multihead_attn(query=query, key=key, value=key)
        query = query + self.dropout1(query1)
        query = self.norm1(query)

        query2 = self.linear2(self.dropout(F.relu(self.linear1(query))))
        query = query + self.dropout2(query2)
        query = self.norm2(query)

        query = query.permute(1, 2, 0)
        return query
