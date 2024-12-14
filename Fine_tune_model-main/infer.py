import torch
from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, GPT2Tokenizer
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt
import argparse
import warnings

warnings.filterwarnings("ignore")

def load_model(model_path, image_encoder_model, text_decoder_model, device):
    model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(
        image_encoder_model, text_decoder_model
    )

    # Load the saved checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    # Set configurations
    tokenizer = GPT2Tokenizer.from_pretrained(text_decoder_model)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.decoder_start_token_id = tokenizer.bos_token_id

    feature_extractor = ViTFeatureExtractor.from_pretrained(image_encoder_model)

    return model, tokenizer, feature_extractor

def preprocess_image(image_path):
    transforms = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),  # Convert to tensor and normalize to [0,1]
    ])
    image = Image.open(image_path).convert("RGB")
    return transforms(image)

def generate_caption(model, feature_extractor, tokenizer, image_tensor, device):
  
    inputs = feature_extractor(image_tensor, return_tensors="pt", do_rescale=False)
    pixel_values = inputs["pixel_values"].to(device)

    # Generate caption
    with torch.no_grad():
        outputs = model.generate(
            pixel_values=pixel_values,
            max_length=20,
            num_beams=8,
            early_stopping=True, 
            pad_token_id = tokenizer.eos_token_id
        )
    caption = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return caption

def main(args):
    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model, tokenizer, and feature extractor
    model, tokenizer, feature_extractor = load_model(
        args.model_path, "google/vit-base-patch16-224-in21k", "gpt2", device
    )

    # Preprocess the image
    image_tensor = preprocess_image(args.image_path)

    # Generate caption
    caption = generate_caption(model, feature_extractor, tokenizer, image_tensor, device)
    print(f"Generated Caption: {caption}")

    # Visualize the image with the generated caption
    plt.figure(figsize=(6, 6))
    plt.imshow(image_tensor.permute(1, 2, 0))
    plt.title(caption)
    plt.axis("off")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Arguments for model and paths
    parser.add_argument("--model_path", type=str, required=True, help="Path to the saved model checkpoint.")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the input image.")

    args = parser.parse_args()
    main(args)
