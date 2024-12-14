from torchsummary import summary
from base_model import DecoderRNN
import torch
import sys
from dataset import * 

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
vocab_path = "flickr30k_vocab.pkl"
vocab = load_vocab(vocab_path)

class DecoderRNNWrapper(DecoderRNN):
    def forward(self, features):
        batch_size = features.size(0)
        seq_length = 10 
        dummy_captions = torch.zeros(batch_size, seq_length, dtype=torch.long).to(features.device)
        return super().forward(features, dummy_captions)
    
# Define input size for models
decoder_input_size = (49, 2048)  
vocab_size = len(vocab)  
embed_size = 256
attention_dim = 256
decoder_dim = 512

decoder = DecoderRNNWrapper(embed_size, vocab_size, attention_dim, 2048, decoder_dim).to(device)

decoder_file = "decoder_summary.txt"

def save_summary(model, input_size, file_path):
    with open(file_path, 'w') as f:
        sys.stdout = f  
        summary(model, input_size=input_size)
    sys.stdout = sys.__stdout__  

save_summary(decoder, input_size=(49, 2048), file_path=decoder_file)
