import torch
from model.Attentionbased.Encoder import Encoder
from model.Attentionbased.Decoder import Decoder
from model.Attentionbased.Attention import AttentionMechanism
from model.Attentionbased.AttentionbasedSeq2seq import AttnSeq2Seq
from train import preprocess_sentence
import pickle

def translate_sentence(model, sentence, en_vocab, vi_vocab, max_length=20, device='cpu'):
    # Preprocess the input sentence
    sentence = preprocess_sentence(sentence, 'en')
    
    # Chuyển đổi câu đầu vào thành chỉ số (tokenize)
    inputs = [en_vocab.word2index.get(word, en_vocab.word2index['<unk>']) for word in sentence.split(' ')]
    inputs = torch.tensor(inputs, dtype=torch.long).unsqueeze(0).to(device)  # Thêm batch dimension
    
    # Bắt đầu inference
    result = []
    hidden = model.encoder.init_hidden(1)  # Batch size = 1
    encoder_output, encoder_hidden = model.encoder(inputs, hidden)
    
    # Bắt đầu với token <start>
    decoder_input = torch.tensor([vi_vocab.word2index['<start>']], dtype=torch.long).to(device)
    decoder_hidden = encoder_hidden
    
    for t in range(max_length):
        decoder_output, decoder_hidden, _ = model.decoder(decoder_input, decoder_hidden, encoder_output)
        _, topi = decoder_output.data.topk(1)  # Chọn từ có xác suất cao nhất
        
        if topi.item() == vi_vocab.word2index['<end>']:
            break
        
        result.append(topi.item())
        decoder_input = topi.squeeze().detach()
    
    translated_sentence = ' '.join([vi_vocab.index2word[idx] for idx in result])
    return translated_sentence

# Function to load the model from a checkpoint
def load_checkpoint(checkpoint_path, encoder, decoder, encoder_optimizer=None, decoder_optimizer=None):
    checkpoint = torch.load(checkpoint_path)
    
    # Load model weights
    encoder.load_state_dict(checkpoint['encoder'])
    decoder.load_state_dict(checkpoint['decoder'])
    
    # Optionally load the optimizer states if you want to resume training
    if encoder_optimizer:
        encoder_optimizer.load_state_dict(checkpoint['encoder_optimizer'])
    if decoder_optimizer:
        decoder_optimizer.load_state_dict(checkpoint['decoder_optimizer'])
    
    print(f"Checkpoint loaded from '{checkpoint_path}'")

with open('en_vocab.pkl', 'rb') as f:
    en_vocab = pickle.load(f)

with open('vi_vocab.pkl', 'rb') as f:
    vi_vocab = pickle.load(f)

# Khởi tạo các thông số của mô hình
vocab_enc_size = len(en_vocab.word2index)  # Kích thước từ điển tiếng Anh
embedding_dim = 256                        # Kích thước nhúng
units = 1024                               # Số lượng units trong các lớp ẩn
n_layers = 2                               # Số lượng tầng của encoder/decoder
vocab_dec_size = len(vi_vocab.word2index)  # Kích thước từ điển tiếng Việt
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Khởi tạo mô hình encoder, decoder và attention
encoder = Encoder(vocab_enc_size, embedding_dim, units, n_layers).to(device)
decoder = Decoder(vocab_dec_size, embedding_dim, units, n_layers).to(device)
attention = AttentionMechanism(units)
model = AttnSeq2Seq(encoder, decoder, attention).to(device)

# Load checkpoint
checkpoint_path = 'checkpoint_epoch_10.pt'
load_checkpoint(checkpoint_path, encoder, decoder)

# Dịch câu mới
sentence = "I love programming."
translated_sentence = translate_sentence(model, sentence, en_vocab, vi_vocab, device=device)
print(f"Translated: {translated_sentence}")