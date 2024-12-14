import torch
from PIL import Image
import torchvision.transforms as T
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from collections import namedtuple
from base_model import EncoderDecoder
from dataset import *
import random
from torch.utils.data import Subset
import nltk

greedy_model_path = "base_model.pth"
beam_model_path = "new_beam_search.pth"
vocab_path = 'flickr30k_vocab.pkl'

captions_file = "D:/HUST/20241/Intro to deep learning/flickr8k/captions.txt"  
root_dir = "D:/HUST/20241/Intro to deep learning/flickr8k/Images/"  
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

transform = T.Compose([T.Resize(226), T.RandomCrop(224), T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

dataset = FlickrDataset(root_dir=root_dir, captions_file=captions_file, transform=transform)
random_indices = random.sample(range(len(dataset)), 5000)
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

def greedy_generate_caption(image_path, model, vocab, max_len=20):
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)  
    with torch.no_grad():
        features = model.encoder(image_tensor)
    generated_caption, alphas = model.decoder.greedy_generate_caption(features, max_len=max_len, vocab=vocab)
    generated_caption = [item for item in generated_caption if item not in ('<SOS>', '<EOS>','<UNK>','<PAD>')]
    return  generated_caption

def beam_generate_caption(image_path, model, vocab, max_len=20):
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)  
    with torch.no_grad():
        features = model.encoder(image_tensor)
    generated_caption, alphas = model.decoder.beam_generate_caption(features, max_len=max_len, vocab=vocab)
    generated_caption = [item for item in generated_caption if item not in ('<SOS>', '<EOS>','<UNK>','<PAD>')]
    return  generated_caption

def greedy_bleu_score_checker(model, vocab):
    gc = []
    test = []
    
    for i in range(0,len(dataloader),5):
        temp_gc = []
        model.eval()
        with torch.no_grad():
            img, caption, img_path = dataset.evaluation(i)
            generated_caption = greedy_generate_caption(img_path, model, vocab, max_len=20)
            test.append(generated_caption)
            temp_gc.append(caption)
        gc.append(temp_gc)
        model.train()
    
    print("Greedy nltk metrics")
    BLEU4 = nltk.translate.bleu_score.corpus_bleu(gc, test,weights=(0.25,0.25,0.25,0.25))
    BLEU1 = nltk.translate.bleu_score.corpus_bleu(gc, test,weights=(1.0,0,0,0))
    BLEU2 = nltk.translate.bleu_score.corpus_bleu(gc, test,weights=(0.5,0.5,0,0))
    BLEU3 = nltk.translate.bleu_score.corpus_bleu(gc, test,weights=(0.33,0.33,0.33,0))
    
    
    print(f"BLEU-1 {BLEU1}")
    print(f"BLEU-2 {BLEU2}")
    print(f"BLEU-3 {BLEU3}")
    print(f"BLEU-4 {BLEU4}")

def beam_bleu_score_checker(model, vocab):
    gc = []
    test = []
    
    for i in range(0,len(dataloader),5):
        temp_gc = []
        model.eval()
        with torch.no_grad():
            img, caption, img_path = dataset.evaluation(i)
            generated_caption = beam_generate_caption(img_path, model, vocab, max_len=20)
            test.append(generated_caption)
            temp_gc.append(caption)
        gc.append(temp_gc)
        model.train()
    
    print("Beam search nltk metrics")
    BLEU4 = nltk.translate.bleu_score.corpus_bleu(gc, test,weights=(0.25,0.25,0.25,0.25))
    BLEU1 = nltk.translate.bleu_score.corpus_bleu(gc, test,weights=(1.0,0,0,0))
    BLEU2 = nltk.translate.bleu_score.corpus_bleu(gc, test,weights=(0.5,0.5,0,0))
    BLEU3 = nltk.translate.bleu_score.corpus_bleu(gc, test,weights=(0.33,0.33,0.33,0))
    
    
    print(f"BLEU-1 {BLEU1}")
    print(f"BLEU-2 {BLEU2}")
    print(f"BLEU-3 {BLEU3}")
    print(f"BLEU-4 {BLEU4}")

def main():
    vocab = load_vocab(vocab_path)
    greedy_model = load_model(greedy_model_path, vocab_size=len(vocab), embed_size=300, attention_dim=256, encoder_dim=2048, decoder_dim=512)
    beam_model = load_model(beam_model_path, vocab_size=len(vocab), embed_size=300, attention_dim=256, encoder_dim=2048, decoder_dim=512)

    greedy_bleu_score_checker(greedy_model, vocab)
    print("----------------------------")
    beam_bleu_score_checker(beam_model, vocab)

main()