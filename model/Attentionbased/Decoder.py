import torch.nn as nn
from torch.nn import functional as F
from model.Attentionbased.Attention import AttentionMechanism
import torch

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hidden_dim, n_layers, dropout=0.5):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim + hidden_dim, hidden_dim, n_layers, dropout=dropout, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.attention = AttentionMechanism(hidden_dim)
    
    def forward(self, input, hidden, cell, encoder_outputs):
        embedded = self.dropout(self.embedding(input))
        context_vector, _ = self.attention(hidden[-1], encoder_outputs)
        context_vector = context_vector.unsqueeze(1)
        rnn_input = torch.cat((embedded, context_vector), dim=2)
        output, (hidden, cell) = self.rnn(rnn_input, (hidden, cell))
        prediction = self.fc_out(output)
        return prediction, hidden, cell