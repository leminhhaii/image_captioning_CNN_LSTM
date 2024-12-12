import nltk
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
from nltk.translate.bleu_score import corpus_bleu
from dataset import *
from model import EncoderDecoder
from torch.utils.data import Subset
import random
import matplotlib.pyplot as plt
# This is a proper BLEU score evaluator.

model_path = "new_file/attention_model_state.pth" 
captions_file = "flickr8k/captions.txt"  
root_dir = "flickr8k/Images/"  
vocab_path = "new_file/flickr30k_vocab.pkl"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

vocab = load_vocab(vocab_path)

transform = T.Compose([T.Resize(256), 
                    T.CenterCrop(224), 
                    T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

dataset = FlickrDataset(root_dir=root_dir, captions_file=captions_file, transform=transform)
random_indices = random.sample(range(len(dataset)), 20)
subset_dataset = Subset(dataset, random_indices)

dataloader = DataLoader(subset_dataset, batch_size=1, shuffle=False)

def load_model(model_path, vocab_size, embed_size, attention_dim, encoder_dim, decoder_dim):
    model = EncoderDecoder(
        embed_size=embed_size,
        vocab_size=vocab_size,
        attention_dim=attention_dim,
        encoder_dim=encoder_dim,
        decoder_dim=decoder_dim
    ).to(device)

    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    return model

model = load_model(model_path, vocab_size=len(vocab), embed_size=300, attention_dim=256, encoder_dim=2048, decoder_dim=512)

def bleu_score_checker():
    
    gc = []
    test = []
    
    for i in range(0,len(dataloader),5):
        temp_gc = []
        model.eval()
        with torch.no_grad():
            img, caption = dataset.evaluation(i)
            img = img.unsqueeze(0).to(device)
            encoded_output = model.encoder(img)
            generated_caption, alphas = model.decoder.generate_caption(encoded_output,max_len=20,vocab=vocab)
            test.append(generated_caption)
            temp_gc.append(caption)
        gc.append(temp_gc)
        model.train()
    
    print("Nltk metrics")
    BLEU4 = nltk.translate.bleu_score.corpus_bleu(gc, test,weights=(0.25,0.25,0.25,0.25))
    BLEU1 = nltk.translate.bleu_score.corpus_bleu(gc, test,weights=(1.0,0,0,0))
    BLEU2 = nltk.translate.bleu_score.corpus_bleu(gc, test,weights=(0.5,0.5,0,0))
    BLEU3 = nltk.translate.bleu_score.corpus_bleu(gc, test,weights=(0.33,0.33,0.33,0))
    
    
    print(f"BLEU-1 {BLEU1}")
    print(f"BLEU-2 {BLEU2}")
    print(f"BLEU-3 {BLEU3}")
    print(f"BLEU-4 {BLEU4}")
    
    # print("GC" , gc)
    # print("Predictions", test)
        
bleu_score_checker()