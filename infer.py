import torch
import pickle
import numpy as np
from PIL import Image
import torchvision.transforms as T
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
from torchvision import models
from collections import namedtuple
from model import EncoderDecoder
from collections import Counter
from dataset import *
import random
from torch.utils.data import Subset
import nltk

model_path = 'output/attention_model_state.pth'
vocab_path = 'output/flickr30k_vocab.pkl'

captions_file = "D:/HUST/20241/Intro to deep learning/flickr8k/captions.txt"  
root_dir = "D:/HUST/20241/Intro to deep learning/flickr8k/Images/"  
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

transform = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

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


def show_image_and_caption(image_path, caption):
    img = Image.open(image_path)
    plt.imshow(img)
    plt.title(caption)
    plt.axis('off')
    plt.show()

#Show attention
def plot_attention(img, result, attention_plot):
    #untransform
    img[0] = img[0] * 0.229
    img[1] = img[1] * 0.224 
    img[2] = img[2] * 0.225 
    img[0] += 0.485 
    img[1] += 0.456 
    img[2] += 0.406
    
    img = img.numpy().transpose((1, 2, 0))
    temp_image = img

    fig = plt.figure(figsize=(15, 15))

    len_result = len(result)
    for l in range(len_result):
        temp_att = attention_plot[l].reshape(7,7)
        
        ax = fig.add_subplot(len_result//2,len_result//2, l+1)
        ax.set_title(result[l])
        img = ax.imshow(temp_image)
        ax.imshow(temp_att, cmap='gray', alpha=0.7, extent=img.get_extent())
        

    plt.tight_layout()
    plt.show()


# Inference function to generate captions for an image
def generate_caption(image_path, model, vocab, max_len=20):
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)  

    with torch.no_grad():
        features = model.encoder(image_tensor)

    # Generate the caption using the decoder
    generated_caption, alphas = model.decoder.generate_caption(features, max_len=max_len, vocab=vocab)
    # img1 = image_tensor[0].detach().clone()
    # plot_attention(img1, generated_caption, alphas)
    return ' '.join(generated_caption), generated_caption

def bleu_score_checker(model, vocab):
    gc = []
    test = []
    
    for i in range(0,len(dataloader),5):
        temp_gc = []
        model.eval()
        with torch.no_grad():
            img, caption, img_path = dataset.evaluation(i)
            _, generated_caption = generate_caption(img_path, model, vocab, max_len=20)
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

def caption_predict(image_path, model, vocab):
    caption,_ = generate_caption(image_path, model, vocab)
    show_image_and_caption(image_path, caption)

def main():
    vocab = load_vocab(vocab_path)
    model = load_model(model_path, vocab_size=len(vocab), embed_size=300, attention_dim=256, encoder_dim=2048, decoder_dim=512)
    image_path ="C:/Users/Admin/Desktop/Screenshot 2024-12-12 214057.png"
    # Example: Generate a caption for a new image
    # image_path = 'flickr8k/Images/10815824_2997e03d76.jpg'  
    
    # caption_predict(image_path, model, vocab)
    bleu_score_checker(model, vocab)
main()
