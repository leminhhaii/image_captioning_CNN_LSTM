import os
from collections import Counter
import spacy
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader,Dataset
import torchvision.transforms as T
import pickle
from PIL import Image
import pandas as pd

spacy_eng = spacy.load("en_core_web_sm")

class Vocabulary:
    def __init__(self,freq_threshold):
        self.itos = {0:"<PAD>",1:"<SOS>",2:"<EOS>",3:"<UNK>"}
        self.stoi = {v:k for k,v in self.itos.items()}
        self.freq_threshold = freq_threshold
        
    def __len__(self): return len(self.itos)
    
    @staticmethod
    def tokenize(text):
        return [token.text.lower() for token in spacy_eng.tokenizer(str(text))]
    
    def build_vocab(self, sentence_list):
        frequencies = Counter()
        idx = 4
        
        for sentence in sentence_list:
            for word in self.tokenize(sentence):
                frequencies[word] += 1
                
                if frequencies[word] == self.freq_threshold:
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1
    
    def numericalize(self,text):
        tokenized_text = self.tokenize(text)
        return [ self.stoi[token] if token in self.stoi else self.stoi["<UNK>"] for token in tokenized_text ]  

    def save_vocab(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump({'itos': self.itos, 'stoi': self.stoi}, f)

def load_vocab(filepath):
    with open(filepath, 'rb') as f:
        vocab_data = pickle.load(f)
    vocab = Vocabulary(freq_threshold=5)
    vocab.itos = vocab_data['itos']
    vocab.stoi = vocab_data['stoi']
    return vocab

class FlickrDataset(Dataset):
    def __init__(self,root_dir,captions_file,transform=None,freq_threshold=5):
        self.root_dir = root_dir
        self.df = pd.read_csv(captions_file)
        self.transform = transform
        
        self.imgs = self.df["image"]
        self.captions = self.df["caption"]
        
        self.vocab = Vocabulary(freq_threshold)
        self.vocab.build_vocab(self.captions.tolist())

        
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self,idx: int):
        caption = self.captions.iloc[idx]
        img_id = self.imgs.iloc[idx]
        
        
        img_location = os.path.join(self.root_dir,img_id)
        img = Image.open(img_location).convert("RGB")
        
        if self.transform is not None:
            img = self.transform(img)
        
        caption_vec = []
        caption_vec += [self.vocab.stoi["<SOS>"]]
        caption_vec += self.vocab.numericalize(caption)
        caption_vec += [self.vocab.stoi["<EOS>"]]
    
        return img, torch.tensor(caption_vec)
    
    def evaluation(self, idx):
        caption = self.captions[idx]
        img_id = self.imgs[idx]

        img_path = os.path.join(self.root_dir, img_id)
        img = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)
        
        caption = self.vocab.tokenize(caption)
        
        return img, caption, img_path
