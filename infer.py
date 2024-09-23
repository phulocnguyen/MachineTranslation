import torch
from model.Attentionbased.Encoder import Encoder
from model.Attentionbased.Decoder import Decoder
from model.Attentionbased.Attention import AttentionMechanism
from model.Attentionbased.AttentionbasedSeq2seq import AttnSeq2Seq
from train import *
import pickle

import torch

def preprocess_input_sentence(sentence, vocab):
    """Preprocesses the input sentence to match the format used during training."""
    sentence = preprocess_sentence(sentence, lang='en')
    tensor = torch.tensor([vocab.word2index.get(word, vocab.word2index['<unk>']) for word in sentence.split()])
    return tensor.unsqueeze(0)  # Add batch dimension

def translate_sentence(model, sentence, en_vocab, vi_vocab, max_length=50):
    """Translates a single input sentence from English to Vietnamese."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Preprocess the input sentence
    input_tensor = preprocess_input_sentence(sentence, en_vocab).to(device)
    
    # Initialize the decoder's input with the <start> token and set hidden states
    start_token = torch.tensor([vi_vocab.word2index['<start>']]).unsqueeze(0).to(device)
    end_token = vi_vocab.word2index['<end>']
    
    hidden_states = None
    translation = []
    
    # Run through the encoder
    encoder_output, hidden_states = model.encoder(input_tensor)
    
    # Start decoding
    decoder_input = start_token
    attention_weights = []

    for t in range(max_length):
        # Run one step through the decoder
        decoder_output, hidden_states, attn_weights = model.decoder(
            decoder_input, hidden_states, encoder_output)  # Pass encoder_output here!
        
        attention_weights.append(attn_weights)
        
        # Select the word with the highest probability
        predicted_token = decoder_output.argmax(dim=1).item()

        # Stop if <end> token is reached
        if predicted_token == end_token:
            break
        
        # Append the predicted word to the translation
        translation.append(predicted_token)
        
        # The predicted token becomes the input for the next step
        decoder_input = torch.tensor([[predicted_token]]).to(device)

    # Convert the token indices back to words
    translated_sentence = ' '.join([vi_vocab.index2word[tok] for tok in translation])
    return translated_sentence


def infer():
    # Load your trained model and vocabularies
    path_en = "/Users/phulocnguyen/Documents/Workspace/Machine Translation/dataset/train-en-vi/train.en"
    path_vi = "/Users/phulocnguyen/Documents/Workspace/Machine Translation/dataset/train-en-vi/train.vi"
    num_examples = 100
    batch_size = 32
    
    # Recreate data and vocab
    _, _, en_vocab, vi_vocab = load_data(path_en, path_vi, batch_size=batch_size, num_examples=num_examples)
    
    embedding_dim = 256
    units = 1024
    n_layers = 2
    vocab_enc_size = len(en_vocab.word2index)
    vocab_dec_size = len(vi_vocab.word2index)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = Encoder(vocab_enc_size, embedding_dim, units, n_layers).to(device)
    decoder = Decoder(vocab_dec_size, embedding_dim, units, n_layers).to(device)
    attention = AttentionMechanism(units)
    model = AttnSeq2Seq(encoder, decoder, attention).to(device)
    
    # Load the model checkpoint
    checkpoint = torch.load('checkpoint_epoch_10.pt', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Set the model to evaluation mode
    model.eval()

    # Translate an example sentence
    input_sentence = "How are you?"
    translated_sentence = translate_sentence(model, input_sentence, en_vocab, vi_vocab)
    
    print(f"Input: {input_sentence}")
    print(f"Translated: {translated_sentence}")

# Run the inference
infer()
