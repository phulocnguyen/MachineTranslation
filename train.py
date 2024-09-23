import unicodedata
import re
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence
from collections import Counter
import os
from model.Attentionbased.Encoder import Encoder
from model.Attentionbased.Decoder import Decoder
from model.Attentionbased.Attention import AttentionMechanism
from model.Attentionbased.AttentionbasedSeq2seq import AttnSeq2Seq
import time
import torch.nn as nn
import torch.optim as optim
import pickle

def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn')

def preprocess_sentence(w, lang):
    w = unicode_to_ascii(w.lower().strip())
    
    # Tạo khoảng trắng giữa từ và dấu câu
    w = re.sub(r"([?.!,¿])", r" \1 ", w)
    w = re.sub(r'[" "]+', " ", w)
    
    # Đối với tiếng Anh, loại bỏ các ký tự không phải a-z và dấu câu
    if lang == 'en':
        w = re.sub(r"[^a-zA-Z?.!,¿]+", " ", w)
    
    w = w.strip()
    
    # Thêm token bắt đầu và kết thúc
    w = '<start> ' + w + ' <end>'
    return w

def create_dataset(path_en, path_vi, num_examples=None):
    lines_en = open(path_en, encoding='UTF-8').read().strip().split('\n')
    lines_vi = open(path_vi, encoding='UTF-8').read().strip().split('\n')
    
    word_pairs = []
    for en, vi in zip(lines_en[:num_examples], lines_vi[:num_examples]):
        en = preprocess_sentence(en, 'en')
        vi = preprocess_sentence(vi, 'vi')
        word_pairs.append([en, vi])
    
    return zip(*word_pairs)

class Vocab:
    def __init__(self, lang):
        self.lang = lang
        self.word2index = {'<pad>': 0, '<start>': 1, '<end>': 2, '<unk>': 3}
        self.word2count = {'<pad>': 1, '<start>': 1, '<end>': 1, '<unk>': 1}
        self.index2word = {0: '<pad>', 1: '<start>', 2: '<end>', 3: '<unk>'}
        self.n_words = 4  # Count default tokens

    def add_sentence(self, sentence):
        for word in sentence.split(' '):
            self.add_word(word)

    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

class TranslationDataset(Dataset):
    def __init__(self, en_sentences, vi_sentences):
        self.en_sentences = en_sentences
        self.vi_sentences = vi_sentences
        
        self.en_vocab = Vocab('en')
        self.vi_vocab = Vocab('vi')
        
        for en, vi in zip(en_sentences, vi_sentences):
            self.en_vocab.add_sentence(en)
            self.vi_vocab.add_sentence(vi)

    def __len__(self):
        return len(self.en_sentences)

    def __getitem__(self, idx):
        en_sentence = self.en_sentences[idx]
        vi_sentence = self.vi_sentences[idx]
        
        en_tensor = torch.tensor([self.en_vocab.word2index.get(word, self.en_vocab.word2index['<unk>']) for word in en_sentence.split()])
        vi_tensor = torch.tensor([self.vi_vocab.word2index.get(word, self.vi_vocab.word2index['<unk>']) for word in vi_sentence.split()])
        
        return en_tensor, vi_tensor

def collate_fn(batch):
    en_batch, vi_batch = zip(*batch)
    en_batch = pad_sequence(en_batch, batch_first=True, padding_value=0)
    vi_batch = pad_sequence(vi_batch, batch_first=True, padding_value=0)
    return en_batch, vi_batch

def load_data(path_en, path_vi, batch_size=32, num_examples=None, test_size=0.3):
    en_sentences, vi_sentences = create_dataset(path_en, path_vi, num_examples)
    dataset = TranslationDataset(en_sentences, vi_sentences)
    
    # Tính toán kích thước của tập train và validation
    val_size = int(len(dataset) * test_size)
    train_size = len(dataset) - val_size
    
    # Chia dataset thành tập train và validation
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Tạo DataLoader cho tập train và validation
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    return train_dataloader, val_dataloader, dataset.en_vocab, dataset.vi_vocab

def max_length(tensor):
    return max(len(t) for t in tensor)


# # Sử dụng


# with open('en_vocab.pkl', 'wb') as f:
#     pickle.dump(en_vocab, f)

# with open('vi_vocab.pkl', 'wb') as f:
#     pickle.dump(vi_vocab, f)

# # Tính toán max_length


# print(f"# training data: {len(train_dataloader.dataset)}")
# print(f"# validation data: {len(val_dataloader.dataset)}")



# Initialize models
def train_step(inp, targ, model, optimizer, criterion, teacher_forcing_ratio=0.5):
    optimizer.zero_grad()
    
    # Forward pass qua model (AttnSeq2Seq)
    output = model(inp, targ, teacher_forcing_ratio)
    
    # Reshape lại output và target để tính loss
    output = output.view(-1, output.shape[-1])  # [batch_size * trg_len, vocab_size]
    targ = targ.view(-1)  # [batch_size * trg_len]
    
    # Tính loss
    loss = criterion(output, targ)
    
    # Backpropagation và update weights
    loss.backward()
    optimizer.step()
    
    return loss.item()

def save_checkpoint(model, checkpoint_path):
    torch.save({
        'model_state_dict': model.state_dict(),
    }, checkpoint_path)

def training_loop(train_dataloader, model, num_epochs, optimizer, criterion):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Training loop
    for epoch in range(num_epochs):
        start = time.time()
        total_loss = 0

        for batch, (inp, targ) in enumerate(train_dataloader):
            inp, targ = inp.to(device), targ.to(device)
            batch_loss = train_step(inp, targ, model, optimizer, criterion)
            total_loss += batch_loss

            if batch % 1 == 0:
                print(f'Epoch {epoch + 1} Batch {batch} Loss {batch_loss:.4f}')

        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            save_checkpoint(model, f'checkpoint_epoch_{epoch+1}.pt')

        print(f'Epoch {epoch + 1} Loss {total_loss / len(train_dataloader):.4f}')
        print(f'Time taken for 1 epoch {time.time() - start:.2f} sec\n')

def main():
    # Cấu hình cho quá trình training
    path_en = "/Users/phulocnguyen/Documents/Workspace/Machine Translation/dataset/train-en-vi/train.en"
    path_vi = "/Users/phulocnguyen/Documents/Workspace/Machine Translation/dataset/train-en-vi/train.vi"
    num_examples = 100
    batch_size = 32

    train_dataloader, val_dataloader, en_vocab, vi_vocab = load_data(path_en, path_vi, batch_size=batch_size, num_examples=num_examples)
    max_length_targ = max(max_length(batch[1]) for batch in train_dataloader)
    max_length_inp = max(max_length(batch[0]) for batch in train_dataloader)
    BATCH_SIZE = batch_size
    steps_per_epoch = len(train_dataloader)
    embedding_dim = 256
    units = 1024
    n_layers = 2
    vocab_enc_size = len(en_vocab.word2index)
    vocab_dec_size = len(vi_vocab.word2index)
    
    learning_rate = 1e-3
    EPOCHS = 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = Encoder(vocab_enc_size, embedding_dim, units, n_layers).to(device)
    decoder = Decoder(vocab_dec_size, embedding_dim, units, n_layers).to(device)
    attention = AttentionMechanism(units)
    model = AttnSeq2Seq(encoder, decoder, attention).to(device)

    # Optimizers
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Loss function
    criterion = nn.CrossEntropyLoss(ignore_index=0, reduction='mean')
    training_loop(train_dataloader=train_dataloader, model=model, num_epochs=EPOCHS, optimizer=optimizer, criterion=criterion)

main()