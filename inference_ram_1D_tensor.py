'''
 * The Recognize Anything Plus Model (RAM++)
 * Written by Xinyu Huang
 * Update the code to output 1D tensor by Phongsiri Nirachornkul
'''
import argparse
import numpy as np
import torch
from PIL import Image
from datetime import datetime
import os
from ram.models import ram_plus
from ram import inference_ram as inference
from ram import get_transform

# Argument parsing
parser = argparse.ArgumentParser(description='Tag2Text inference for tagging and captioning')
parser.add_argument('--image', metavar='DIR', help='path to dataset', default='images/demo/demo1.jpg')
parser.add_argument('--pretrained', metavar='DIR', help='path to pretrained model', default='pretrained/ram_plus_swin_large_14m.pth')
parser.add_argument('--image-size', default=384, type=int, metavar='N', help='input image size (default: 448)')

if __name__ == "__main__":
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Image transformation
    transform = get_transform(image_size=args.image_size)

    # Load model
    model = ram_plus(pretrained=args.pretrained, image_size=args.image_size, vit='swin_l')
    model.eval()
    model = model.to(device)

    # Load and transform image
    image = transform(Image.open(args.image)).unsqueeze(0).to(device)

    # Forward pass to get latent features
    latent_features = model.extract_latent_features(image)
    latent_features_1d = latent_features.view(-1)  # Convert to 1D tensor

    # Output latent features
    print("Latent Features (1D tensor):", latent_features_1d.cpu().numpy())

    # Create output directory if it doesn't exist
    output_dir = "output/1D_tensor"
    os.makedirs(output_dir, exist_ok=True)

    # Save latent features to a text file with timestamp
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = os.path.join(output_dir, f"latent_features_{current_time}.txt")
    np.savetxt(file_name, latent_features_1d.cpu().numpy(), fmt='%.6f')

    print(f"Latent features saved to {file_name}")

    # Run inference
    res = inference(image, model)
    print("Image Tags: ", res[0])
    print("图像标签: ", res[1])
