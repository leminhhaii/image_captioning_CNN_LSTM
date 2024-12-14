import torch
from PIL import Image
import torchvision.transforms as T
import matplotlib.pyplot as plt
from model import EncoderDecoder
from dataset import *

MODEL_FILE = "glove_model.pth"
VOCAB_FILE = "flickr30k_vocab.pkl"
glove_files = "glove.6B.300d.txt"
# Replace with your image directory in your computer
image_path ="D:/HUST/20241/Intro to deep learning/flickr8k/Images/385835044_4aa11f6990.jpg"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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

def show_image_and_caption(image_path, caption):
    img = Image.open(image_path)
    plt.imshow(img)
    plt.title(caption)
    plt.axis('off')
    plt.show()

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

vocab = load_vocab(VOCAB_FILE)
word_to_idxs = vocab.stoi
model = load_model(MODEL_FILE, vocab_size=len(vocab), embed_size=300, attention_dim=256, encoder_dim=2048, decoder_dim=512, glove_file = glove_files, word_to_idx = word_to_idxs)

caption,_ = generate_caption(image_path, model, vocab)
show_image_and_caption(image_path, caption)