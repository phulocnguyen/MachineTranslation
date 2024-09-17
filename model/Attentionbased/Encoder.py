import torch.nn as nn
from torch.nn import functional as F

class Encoder(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, n_layers, dropout_probs=0.5):
        super(Encoder, self).__init__()
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.embed_layer = nn.Embedding(input_dim, embedding_dim)
        self.LSTM = nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout=dropout_probs, batch_first=True)
        self.drop_out = nn.Dropout(p=dropout_probs)
    
    def forward(self, input):
        embedded = self.drop_out(self.embed_layer(input))
        output, (hidden, cell) = self.LSTM(embedded)
        return output, (hidden, cell)