import torch
import torch.nn.functional as F
from PIL import Image
from transformers import ViTFeatureExtractor, ViTModel
import matplotlib.pyplot as plt
import cv2
import numpy as np

# Preprocess image

def preprocess_image(observation: np.ndarray) -> np.ndarray:

    # Resize to (80, 60)
    observation = cv2.resize(observation, (80, 60), interpolation=cv2.INTER_AREA)

    # Transpose to channel-first format
    #observation = np.transpose(observation, (2, 0, 1)).astype(np.float32)

    return observation

# Load the image
def load_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = image.resize((224, 224))  # Resize to match ViT model's expected input size

    return image

def plot_all_attention(image, attentions, head_idx=0):
    """
    Plot attention maps for all layers for a specific head.
    """
    num_layers = len(attentions)  # Total number of layers
    fig, axes = plt.subplots(1, num_layers, figsize=(20, 5))  # One subplot per layer

    for layer_idx in range(num_layers):
        # Extract the attention map for the specific layer and head
        attention_map = attentions[layer_idx][0, head_idx, 1:, 1:].detach().cpu().unsqueeze(0)

        # Downsample to 14x14 using average pooling
        attention_map = F.adaptive_avg_pool2d(attention_map, (14, 14))
        attention_map = attention_map.squeeze(0)  # Remove batch dimension

        # Reshape into a square grid (already 14x14 due to pooling)
        attention_map = attention_map.numpy()

        # Resize the original image to match the grid
        resized_image = image.resize((14 * 16, 14 * 16))

        # Plot attention map overlayed on the original image
        axes[layer_idx].imshow(resized_image, alpha=0.5)
        axes[layer_idx].imshow(attention_map, cmap="viridis", alpha=0.7)
        axes[layer_idx].axis("off")
        axes[layer_idx].set_title(f"Layer {layer_idx}")

    plt.tight_layout()
    plt.show()

# Main function to process the image and extract attention maps
def main(image_path):
    # Load pre-trained ViT model and feature extractor
    model_name = "google/vit-base-patch16-224"
    feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)
    model = ViTModel.from_pretrained(model_name, output_attentions=True,attn_implementation="eager")

    # Load the image
    image = load_image(image_path)

    # Preprocess the image
    inputs = feature_extractor(images=image, return_tensors="pt")

    # Forward pass to get the attention weights
    outputs = model(**inputs)
    attentions = outputs.attentions  # List of attention maps for each layer
    #print(attentions[0].shape)

    # Visualize the attention map for the first layer and first head
    plot_all_attention(image, attentions)

# Run the script
if __name__ == "__main__":
    image_path = "/home/cubos98/catkin_ws/src/Vehicle/first_image_received.jpg"  # Replace with your image path
    main(image_path)
