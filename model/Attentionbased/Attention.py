import torch
import torch.nn as nn
from torch.nn import functional as F

class AttentionMechanism(nn.Module):
    def __init__(self, hidden_dim):
        super(AttentionMechanism, self).__init__()
        self.Wa = nn.Linear(hidden_dim, hidden_dim)
        self.Ua = nn.Linear(hidden_dim, hidden_dim)
        self.Va = nn.Linear(hidden_dim, 1)
    
    def forward(self, queries, keys):
        # queries: [batch_size, hidden_dim]
        # keys: [batch_size, seq_len, hidden_dim]

        # Apply linear layers and calculate scores
        scores = self.Va(torch.tanh(self.Wa(queries).unsqueeze(1) + self.Ua(keys)))
        # scores: [batch_size, seq_len, 1]

        # Softmax over the sequence length to get attention weights
        weights = F.softmax(scores, dim=1)
        # weights: [batch_size, seq_len, 1]

        # Compute the context vector as a weighted sum of the keys
        context = torch.bmm(weights.transpose(1, 2), keys)
        # context: [batch_size, 1, hidden_dim]

        # Remove the extra dimension
        context = context.squeeze(1)
        # context: [batch_size, hidden_dim]

        return context, weights