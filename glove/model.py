import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision.models as models
from torch.utils.data import DataLoader,Dataset
import torchvision.transforms as T
import pickle

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class EncoderCNN(nn.Module):
    """
    Encoder.
    """

    def __init__(self, encoded_image_size=14):
        super(EncoderCNN, self).__init__()
        self.enc_image_size = encoded_image_size

        resnet = models.resnet101(pretrained=True)  # pretrained ImageNet ResNet-101

        # Remove linear and pool layers (since we're not doing classification)
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)

        # Resize image to fixed size to allow input images of variable size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))

        self.fine_tune()

    def forward(self, images):
        """
        Forward propagation.

        :param images: images, a tensor of dimensions (batch_size, 3, image_size, image_size)
        :return: encoded images
        """
        out = self.resnet(images)  # (batch_size, 2048, image_size/32, image_size/32)
        out = self.adaptive_pool(out)  # (batch_size, 2048, encoded_image_size, encoded_image_size)
        out = out.permute(0, 2, 3, 1)  # (batch_size, encoded_image_size, encoded_image_size, 2048)
        out = out.view(out.size(0), -1, out.size(-1))
        return out  # batch, 14x14,2048 

    def fine_tune(self, fine_tune=False):
        """
        Allow or prevent the computation of gradients for convolutional blocks 2 through 4 of the encoder.

        :param fine_tune: Allow?
        """
        for p in self.resnet.parameters():
            p.requires_grad = False
        # If fine-tuning, only fine-tune convolutional blocks 2 through 4
        for c in list(self.resnet.children())[5:]:
            for p in c.parameters():
                p.requires_grad = fine_tune
class Attention(nn.Module):
    """
    Attention Network.
    """

    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        """
        :param encoder_dim: feature size of encoded images
        :param decoder_dim: size of decoder's RNN
        :param attention_dim: size of the attention network
        """
        super(Attention, self).__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)  # linear layer to transform encoded image
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)  # linear layer to transform decoder's output
        self.full_att = nn.Linear(attention_dim, 1)  # linear layer to calculate values to be softmax-ed
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)  # softmax layer to calculate weights

    def forward(self, encoder_out, decoder_hidden):
        """
        Forward propagation.

        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :param decoder_hidden: previous decoder output, a tensor of dimension (batch_size, decoder_dim)
        :return: attention weighted encoding, weights
        """
        att1 = self.encoder_att(encoder_out)  # (batch_size, num_pixels, attention_dim)
        att2 = self.decoder_att(decoder_hidden)  # (batch_size, attention_dim)
        att = self.full_att(self.relu(att1 + att2.unsqueeze(1))).squeeze(2)  # (batch_size, num_pixels)
        alpha = self.softmax(att)  # (batch_size, num_pixels)
        attention_weighted_encoding = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)  # (batch_size, encoder_dim)

        return alpha, attention_weighted_encoding
        
#Attention Decoder
class DecoderRNN(nn.Module):
    def __init__(self, embed_size, vocab_size, attention_dim, encoder_dim, decoder_dim, glove_file,word_to_idx, drop_prob=0.3):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.attention_dim = attention_dim
        self.decoder_dim = decoder_dim

        # Tạo lớp Embedding với GloVe trọng số
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.load_glove_embeddings(glove_file, word_to_idx)

        # Các phần còn lại của mô hình
        self.attention = Attention(encoder_dim, decoder_dim, attention_dim)
        self.init_h = nn.Linear(encoder_dim, decoder_dim)
        self.init_c = nn.Linear(encoder_dim, decoder_dim)
        self.lstm_cell = nn.LSTMCell(embed_size + encoder_dim, decoder_dim, bias=True)
        self.f_beta = nn.Linear(decoder_dim, encoder_dim)
        self.fcn = nn.Linear(decoder_dim, vocab_size)
        self.drop = nn.Dropout(drop_prob)

    def load_glove_embeddings(self, glove_file, word_to_idx):
        """
        Tải embeddings GloVe vào lớp Embedding.
        :param glove_file: đường dẫn đến tệp GloVe embeddings
        """
        # Tạo dictionary cho embeddings
        glove_embeddings = {}
        with open(glove_file, 'r', encoding='utf-8') as f:
            for line in f:
                values = line.split()
                word = values[0]
                vector = np.asarray(values[1:], dtype='float32')
                glove_embeddings[word] = vector

        # Tạo ma trận trọng số embeddings cho vocab của bạn
        embedding_matrix = np.zeros((self.vocab_size, self.embedding.embedding_dim))
        
        for word, idx in word_to_idx.items():
            if word in glove_embeddings:
                embedding_matrix[idx] = glove_embeddings[word]
            else:
                # Nếu từ không có trong GloVe, bạn có thể gán vector ngẫu nhiên
                embedding_matrix[idx] = np.random.uniform(-0.1, 0.1, self.embedding.embedding_dim)
        
        # Copy trọng số vào lớp embedding
        self.embedding.weight.data.copy_(torch.from_numpy(embedding_matrix))

    def forward(self, features, captions):
        embeds = self.embedding(captions)
        h, c = self.init_hidden_state(features)
        seq_length = len(captions[0]) - 1
        batch_size = captions.size(0)
        num_features = features.size(1)

        preds = torch.zeros(batch_size, seq_length, self.vocab_size).to(device)
        alphas = torch.zeros(batch_size, seq_length, num_features).to(device)

        for s in range(seq_length):
            alpha, context = self.attention(features, h)
            lstm_input = torch.cat((embeds[:, s], context), dim=1)
            h, c = self.lstm_cell(lstm_input, (h, c))

            output = self.fcn(self.drop(h))

            preds[:, s] = output
            alphas[:, s] = alpha

        return preds, alphas
    
    # using beam search 
    def generate_caption(self, features, max_len=20, vocab=None, beam_size=3):
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
    def __init__(self,embed_size, vocab_size, attention_dim,encoder_dim,decoder_dim,glove_file,word_to_idx,drop_prob=0.2):
        super().__init__()
        self.encoder = EncoderCNN()
        self.decoder = DecoderRNN(
            embed_size=embed_size,
            vocab_size = vocab_size,
            attention_dim=attention_dim,
            encoder_dim=encoder_dim,
            decoder_dim=decoder_dim,
            glove_file = glove_file,
            word_to_idx= word_to_idx
        )
        
    def forward(self, images, captions):
        features = self.encoder(images)
        outputs = self.decoder(features, captions)
        return outputs