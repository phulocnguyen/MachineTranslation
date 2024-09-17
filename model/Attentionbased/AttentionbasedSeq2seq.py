import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
from model.Attentionbased.Encoder import Encoder
from model.Attentionbased.Decoder import Decoder
from model.Attentionbased.Attention import AttentionMechanism

# class Encoder(nn.Module):
#     def __init__(self, input_dim, embedding_dim, hidden_dim, n_layers, dropout_probs=0.5):
#         super(Encoder, self).__init__()
#         self.input_dim = input_dim
#         self.embedding_dim = embedding_dim
#         self.hidden_dim = hidden_dim
#         self.embed_layer = nn.Embedding(input_dim, embedding_dim)
#         self.LSTM = nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout=dropout_probs, batch_first=True)
#         self.drop_out = nn.Dropout(p=dropout_probs)
    
#     def forward(self, input):
#         embedded = self.drop_out(self.embed_layer(input))
#         output, (hidden, cell) = self.LSTM(embedded)
#         return output, (hidden, cell)

# class Attention(nn.Module):
#     def __init__(self, hidden_dim):
#         super(Attention, self).__init__()
#         self.Wa = nn.Linear(hidden_dim, hidden_dim)
#         self.Ua = nn.Linear(hidden_dim, hidden_dim)
#         self.Va = nn.Linear(hidden_dim, 1)
    
#     def forward(self, queries, keys):
#         # queries: [batch_size, hidden_dim]
#         # keys: [batch_size, seq_len, hidden_dim]

#         # Apply linear layers and calculate scores
#         scores = self.Va(torch.tanh(self.Wa(queries).unsqueeze(1) + self.Ua(keys)))
#         # scores: [batch_size, seq_len, 1]

#         # Softmax over the sequence length to get attention weights
#         weights = F.softmax(scores, dim=1)
#         # weights: [batch_size, seq_len, 1]

#         # Compute the context vector as a weighted sum of the keys
#         context = torch.bmm(weights.transpose(1, 2), keys)
#         # context: [batch_size, 1, hidden_dim]

#         # Remove the extra dimension
#         context = context.squeeze(1)
#         # context: [batch_size, hidden_dim]

#         return context, weights

# class Decoder(nn.Module):
#     def __init__(self, output_dim, emb_dim, hidden_dim, n_layers, dropout=0.5):
#         super(Decoder, self).__init__()
#         self.embedding = nn.Embedding(output_dim, emb_dim)
#         self.rnn = nn.LSTM(emb_dim + hidden_dim, hidden_dim, n_layers, dropout=dropout, batch_first=True)
#         self.fc_out = nn.Linear(hidden_dim, output_dim)
#         self.dropout = nn.Dropout(dropout)
#         self.attention = Attention(hidden_dim)
    
#     def forward(self, input, hidden, cell, encoder_outputs):
#         embedded = self.dropout(self.embedding(input))
#         context_vector, _ = self.attention(hidden[-1], encoder_outputs)
#         context_vector = context_vector.unsqueeze(1)
#         rnn_input = torch.cat((embedded, context_vector), dim=2)
#         output, (hidden, cell) = self.rnn(rnn_input, (hidden, cell))
#         prediction = self.fc_out(output)
#         return prediction, hidden, cell

class AttnSeq2Seq(nn.Module):
    def __init__(self, encoder, decoder, attention):
        super(AttnSeq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.attention = attention
    
    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = src.size(0)
        trg_len = trg.size(1)
        vocab_size = self.decoder.fc_out.out_features
        
        outputs = torch.zeros(batch_size, trg_len, vocab_size).to(src.device)
        encoder_outputs, (hidden, cell) = self.encoder(src)
        input = trg[:, 0]
        
        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(input.unsqueeze(1), hidden, cell, encoder_outputs)
            outputs[:, t] = output.squeeze(1)
            
            top1 = output.argmax(2)
            input = trg[:, t] if torch.rand(1).item() < teacher_forcing_ratio else top1.squeeze(1)
        
        return outputs

# Example usage
input_dim = 1000
emb_dim = 256
hidden_dim = 512
n_layers = 2
output_dim = 1000

encoder = Encoder(input_dim, emb_dim, hidden_dim, n_layers)
decoder = Decoder(output_dim, emb_dim, hidden_dim, n_layers)
attention = AttentionMechanism(hidden_dim)
model = AttnSeq2Seq(encoder, decoder, attention)

# Define optimizer and loss function
optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()

# Example forward pass
src = torch.randint(0, input_dim, (32, 10))  # (batch_size, src_len)
trg = torch.randint(0, output_dim, (32, 20))  # (batch_size, trg_len)
output = model(src, trg)
loss = criterion(output.view(-1, output_dim), trg.view(-1))

print("Output shape:", output.shape)
print("Loss:", loss.item())
