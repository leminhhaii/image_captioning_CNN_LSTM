import torch
from PIL import Image
import torchvision.transforms as T
from torch.utils.data import DataLoader
from model import EncoderDecoder
import random
from torch.utils.data import Subset
import nltk
from dataset import *

MODEL_FILE = "glove_model.pth"
VOCAB_FILE = "flickr30k_vocab.pkl"
glove_files = "glove.6B.300d.txt"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

captions_file = "D:/HUST/20241/Intro to deep learning/flickr8k/captions.txt"  
root_dir = "D:/HUST/20241/Intro to deep learning/flickr8k/Images/"  

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

transform = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

dataset = FlickrDataset(root_dir=root_dir, captions_file=captions_file, transform=transform)
random_indices = random.sample(range(len(dataset)),5000)
subset_dataset = Subset(dataset, random_indices)

dataloader = DataLoader(subset_dataset, batch_size=1, shuffle=False)

def load_model(model_path, vocab_size, embed_size, attention_dim, encoder_dim, decoder_dim, glove_file, word_to_idx):
    model = EncoderDecoder(
        embed_size=embed_size,
        vocab_size=vocab_size,
        attention_dim=attention_dim,
        encoder_dim=encoder_dim,
        decoder_dim=decoder_dim,
        glove_file = glove_file,
        word_to_idx = word_to_idx
    ).to(device)

    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    return model

def generate_caption(image_path, model, vocab, max_len=20, feature_size=(14, 14)):
    image = Image.open(image_path).convert("RGB")
    transform = T.Compose([
        T.Resize(226),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image_tensor = transform(image).unsqueeze(0).to(device)  

    with torch.no_grad():
        features = model.encoder(image_tensor)

    with torch.no_grad():
        generated_caption, alphas = model.decoder.generate_caption(features, max_len=max_len, vocab=vocab)

    generated_caption = [item for item in generated_caption if item not in ('<SOS>', '<EOS>','<UNK>','<PAD>')]
    return ' '.join(generated_caption), generated_caption

def bleu_score_checker(model, vocab):
    gc = []
    test = []
    
    for i in range(0,len(dataloader),5):
        temp_gc = []
        model.eval()
        with torch.no_grad():
            img, caption, img_path = dataset.evaluation(i)
            _,generated_caption = generate_caption(img_path, model, vocab, max_len=20)
            test.append(generated_caption)
            temp_gc.append(caption)
        gc.append(temp_gc)
        model.train()
    
    print("Nltk metrics")
    BLEU4 = nltk.translate.bleu_score.corpus_bleu(gc, test,weights=(0.25,0.25,0.25,0.25))
    BLEU1 = nltk.translate.bleu_score.corpus_bleu(gc, test,weights=(1,0,0,0))
    BLEU2 = nltk.translate.bleu_score.corpus_bleu(gc, test,weights=(0.5,0.5,0,0))
    BLEU3 = nltk.translate.bleu_score.corpus_bleu(gc, test,weights=(0.33,0.33,0.33,0))
    
    print(f"BLEU-1 {BLEU1}")
    print(f"BLEU-2 {BLEU2}")
    print(f"BLEU-3 {BLEU3}")
    print(f"BLEU-4 {BLEU4}")

vocab = load_vocab(VOCAB_FILE)
word_to_idxs = vocab.stoi
model = load_model(MODEL_FILE, vocab_size=len(vocab), embed_size=300, attention_dim=256, encoder_dim=2048, decoder_dim=512, glove_file = glove_files, word_to_idx = word_to_idxs)

bleu_score_checker(model, vocab)