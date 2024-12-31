import torch
from PIL import Image
import torchvision.transforms as T
import matplotlib.pyplot as plt
from base_model import EncoderDecoder
from dataset import *

# Replace with model and vocab file on your computer
greedy_model_path = "base_model.pth"
beam_model_path = "new_beam_search.pth"
vocab_path = 'flickr30k_vocab.pkl'
#Replace with image file direct on your computer
image_path ="D:/HUST/20241/Intro to deep learning/flickr8k/Images/171488318_fb26af58e2.jpg"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

transform = T.Compose([T.Resize(226), T.RandomCrop(224), T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

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

def greedy_generate_caption(image_path, model, vocab, max_len=20):
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)  
    with torch.no_grad():
        features = model.encoder(image_tensor)
    generated_caption, alphas = model.decoder.greedy_generate_caption(features, max_len=max_len, vocab=vocab)
    generated_caption = [item for item in generated_caption if item not in ('<SOS>', '<EOS>','<UNK>','<PAD>')]
    return ' '.join(generated_caption), generated_caption

def beam_generate_caption(image_path, model, vocab, max_len=20):
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)  
    with torch.no_grad():
        features = model.encoder(image_tensor)
    generated_caption, alphas = model.decoder.beam_generate_caption(features, max_len=max_len, vocab=vocab)
    generated_caption = [item for item in generated_caption if item not in ('<SOS>', '<EOS>','<UNK>','<PAD>')]
    return ' '.join(generated_caption), generated_caption
    
def greedy_caption_predict(image_path, model, vocab):
    caption,_ = greedy_generate_caption(image_path, model, vocab)
    show_image_and_caption(image_path, f"Generated caption with greedy search:{caption}")

def beam_caption_predict(image_path, model, vocab):    
    caption,_ = beam_generate_caption(image_path, model, vocab)
    show_image_and_caption(image_path, f"Generated caption with beam search:{caption}")

def main():
    vocab = load_vocab(vocab_path)
    greedy_model = load_model(greedy_model_path, vocab_size=len(vocab), embed_size=300, attention_dim=256, encoder_dim=2048, decoder_dim=512)
    beam_model = load_model(beam_model_path, vocab_size=len(vocab), embed_size=300, attention_dim=256, encoder_dim=2048, decoder_dim=512)
    
    greedy_caption_predict(image_path, greedy_model, vocab)
    beam_caption_predict(image_path, beam_model, vocab)
main()