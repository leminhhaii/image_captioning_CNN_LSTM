import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision.models as models
from torch.utils.data import DataLoader,Dataset
import pickle
import torchvision.transforms as T
# vocab_path = "output/flickr30k_vocab.pkl"
# with open(vocab_path, 'rb') as f:
#     vocab = pickle.load(f)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class EncoderCNN(nn.Module):
    def __init__(self):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet101(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)
        

    def forward(self, images):
        features = self.resnet(images)                                    #(batch_size,2048,7,7)
        features = features.permute(0, 2, 3, 1)                           #(batch_size,7,7,2048)
        features = features.view(features.size(0), -1, features.size(-1)) #(batch_size,49,2048)
        return features

#Bahdanau Attention
class Attention(nn.Module):
    def __init__(self, encoder_dim,decoder_dim,attention_dim):
        super(Attention, self).__init__()
        
        self.attention_dim = attention_dim
        
        self.W = nn.Linear(decoder_dim,attention_dim)
        self.U = nn.Linear(encoder_dim,attention_dim)
        
        self.A = nn.Linear(attention_dim,1)    
        
    def forward(self, features, hidden_state):
        u_hs = self.U(features)     #(batch_size,num_layers,attention_dim)
        w_ah = self.W(hidden_state) #(batch_size,attention_dim)
        
        combined_states = torch.tanh(u_hs + w_ah.unsqueeze(1)) #(batch_size,num_layers,attemtion_dim)
        
        attention_scores = self.A(combined_states)         #(batch_size,num_layers,1)
        attention_scores = attention_scores.squeeze(2)     #(batch_size,num_layers)
        
        alpha = F.softmax(attention_scores,dim=1)          #(batch_size,num_layers)
        
        attention_weights = features * alpha.unsqueeze(2)  #(batch_size,num_layers,features_dim)
        attention_weights = attention_weights.sum(dim=1)   #(batch_size,num_layers)
        
        return alpha,attention_weights
        
#Attention Decoder
class DecoderRNN(nn.Module):
    def __init__(self,embed_size, vocab_size, attention_dim,encoder_dim,decoder_dim,drop_prob=0.3):
        super().__init__()
        
        #save the model param
        self.vocab_size = vocab_size
        self.attention_dim = attention_dim
        self.decoder_dim = decoder_dim
        
        self.embedding = nn.Embedding(vocab_size,embed_size)
        self.attention = Attention(encoder_dim,decoder_dim,attention_dim)
        
        self.init_h = nn.Linear(encoder_dim, decoder_dim)  
        self.init_c = nn.Linear(encoder_dim, decoder_dim)  
        self.lstm_cell = nn.LSTMCell(embed_size+encoder_dim,decoder_dim,bias=True)
        self.f_beta = nn.Linear(decoder_dim, encoder_dim)
        
        self.fcn = nn.Linear(decoder_dim,vocab_size)
        self.drop = nn.Dropout(drop_prob)
    
    def forward(self, features, captions):
        
        #vectorize the caption
        embeds = self.embedding(captions)
        
        # Initialize LSTM state
        h, c = self.init_hidden_state(features)  # (batch_size, decoder_dim)
        
        #get the seq length to iterate
        seq_length = len(captions[0])-1 #Exclude the last one
        batch_size = captions.size(0)
        num_features = features.size(1)
        
        preds = torch.zeros(batch_size, seq_length, self.vocab_size).to(device)
        alphas = torch.zeros(batch_size, seq_length,num_features).to(device)
                
        for s in range(seq_length):
            alpha,context = self.attention(features, h)
            lstm_input = torch.cat((embeds[:, s], context), dim=1)
            h, c = self.lstm_cell(lstm_input, (h, c))
                    
            output = self.fcn(self.drop(h))
            
            preds[:,s] = output
            alphas[:,s] = alpha  
        
        
        return preds, alphas
    
    def greedy_generate_caption(self,features,max_len=20,vocab=None):
        # Inference part
        # Given the image features generate the captions
        
        batch_size = features.size(0)
        h, c = self.init_hidden_state(features)  # (batch_size, decoder_dim)
        
        alphas = []
        
        #starting input
        word = torch.tensor(vocab.stoi['<SOS>']).view(1,-1).to(device)
        embeds = self.embedding(word)
        
        captions = []
        
        for i in range(max_len):
            alpha,context = self.attention(features, h)
            #store the apla score
            alphas.append(alpha.cpu().detach().numpy())
            lstm_input = torch.cat((embeds[:, 0], context), dim=1)
            h, c = self.lstm_cell(lstm_input, (h, c))
            output = self.fcn(self.drop(h))
            output = output.view(batch_size,-1)
            #select the word with most val
            predicted_word_idx = output.argmax(dim=1)
            #save the generated word
            captions.append(predicted_word_idx.item())
            #end if <EOS detected>
            if vocab.itos[predicted_word_idx.item()] == "<EOS>":
                break
            #send generated word as the next caption
            embeds = self.embedding(predicted_word_idx.unsqueeze(0))
        #covert the vocab idx to words and return sentence
        return [vocab.itos[idx] for idx in captions],alphas
    
    # using beam search 
    def beam_generate_caption(self, features, max_len=20, vocab=None, beam_size=3):
        # Inference part using beam search with log-probabilities
        batch_size = features.size(0)
        h, c = self.init_hidden_state(features)  # (batch_size, decoder_dim)
    
        alphas = []
    
        # Starting input (the <SOS> token)
        word = torch.tensor(vocab.stoi['<SOS>']).view(1, -1).to(device)
        embeds = self.embedding(word)
    
        # Initialize beams with the first token and its probability
        beams = [(embeds, h, c, [vocab.stoi['<SOS>']], 0)]  # (input embedding, h, c, list of previous word indices, score)
    
        for step in range(max_len):  # At each timestep
            all_candidates = []
    
            # Expand each beam by generating the next word
            for embeds, h, c, seq, score in beams:  # Iterate over each beam
                # If this beam has finished (contains <EOS>), just add it to all_candidates without generating new word 
                if seq and seq[-1] == vocab.stoi['<EOS>']:
                    all_candidates.append((embeds, h, c, seq, score))
                    continue
    
                # Calculate attention and context for each beam
                alpha, context = self.attention(features, h)
                alphas.append(alpha.cpu().detach().numpy())
    
                lstm_input = torch.cat((embeds[:, 0], context), dim=1)
                h, c = self.lstm_cell(lstm_input, (h, c))
                output = self.fcn(self.drop(h))  # (batch_size, vocab_size)
                output = output.view(batch_size, -1)
    
                # Calculate log-probabilities of the next word
                log_probs = torch.log_softmax(output, dim=1)  # Convert scores to log-probabilities
    
                # Select the top beam_size words with the highest log-probabilities
                topk_log_probs, topk_indices = torch.topk(log_probs, beam_size, dim=1)
    
                for i in range(beam_size):
                    # For each candidate, append the word and score
                    word_idx = topk_indices[:, i]
                    new_log_prob = topk_log_probs[:, i].item()  # Take the log probability of the word
                    new_score = score + new_log_prob  # Add log-probability to the previous score
                    new_seq = seq + [word_idx.item()]
    
                    all_candidates.append((self.embedding(word_idx.unsqueeze(0)), h, c, new_seq, new_score))
    
            # Sort all candidates by score (descending) and update beams
            beams = sorted(all_candidates, key=lambda x: x[4], reverse=True)[:beam_size]
    
        # The beam with the highest score will contain the best caption
        best_beam = beams[0]
        caption = best_beam[3]  # Get the sequence of words with the highest score
    
        # Convert the sequence of word indices to words
        caption_words = [vocab.itos[idx] for idx in caption]
    
        return caption_words, alphas  # caption_words is the generated caption, alphas are the attention weights at each timestep
    
    def init_hidden_state(self, encoder_out):
        mean_encoder_out = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder_out)  # (batch_size, decoder_dim)
        c = self.init_c(mean_encoder_out)
        return h, c
class EncoderDecoder(nn.Module):
    def __init__(self,embed_size, vocab_size, attention_dim,encoder_dim,decoder_dim,drop_prob=0.3):
        super().__init__()
        self.encoder = EncoderCNN()
        self.decoder = DecoderRNN(
            embed_size=embed_size,
            vocab_size = vocab_size,
            attention_dim=attention_dim,
            encoder_dim=encoder_dim,
            decoder_dim=decoder_dim
        )
        
    def forward(self, images, captions):
        features = self.encoder(images)
        outputs = self.decoder(features, captions)
        return outputs
