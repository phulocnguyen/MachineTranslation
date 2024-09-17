from model.Attentionbased.AttentionbasedSeq2seq import AttnSeq2Seq
from utils import translate_sentence

# Load the model and vocab
model = AttnSeq2Seq.load_model('checkpoint_epoch_10.pt')
translated_sentence = translate_sentence(model, "I love learning new languages.", en_vocab, vi_vocab)
print(f"Translation: {translated_sentence}")
