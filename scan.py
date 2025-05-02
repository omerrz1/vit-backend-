import os
import random
import math
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
from transformers import pipeline, AutoImageProcessor, ViTForImageClassification
from captum.attr import LayerGradCam

# ====================== Config ======================
MODEL_PATH = "./trainedModel"
EXPLAIN_DIR = "explainability"
os.makedirs(EXPLAIN_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ====================== Load Model ======================
model = ViTForImageClassification.from_pretrained(MODEL_PATH).to(device).eval()
processor = AutoImageProcessor.from_pretrained(MODEL_PATH, size={"height": 224, "width": 224})

class_names = [
    "Actinic keratosis", "Benign keratosis", "Dermatofibroma",
    "Melanocytic nevus", "Vascular lesion"
]

# ====================== Model Wrapper ======================
class ViTWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(pixel_values=x).logits

# ====================== Grad-CAM Computation ======================
def compute_gradcam(image: Image.Image):
    inputs = processor(image, return_tensors="pt").to(device)
    input_tensor = inputs["pixel_values"]
    input_tensor.requires_grad_()

    # Debug input size
    print("Input tensor shape:", input_tensor.shape)

    outputs = model(pixel_values=input_tensor, output_attentions=True)
    logits = outputs.logits
    pred_class = torch.argmax(logits, dim=-1).item()
    conf = torch.softmax(logits, dim=-1)[0, pred_class].item()

    # Attention map
    attn = outputs.attentions[-1][0].detach().cpu().numpy()  # [heads, tokens, tokens]
    avg_attn = attn.mean(axis=0)  # Average across heads: [tokens, tokens]
    num_tokens = avg_attn.shape[0] - 1  # Exclude CLS token
    grid_size = int(math.ceil(math.sqrt(num_tokens)))
    attn_map = avg_attn[0, 1:].reshape(-1)[:grid_size**2]
    attn_map = np.pad(attn_map, (0, grid_size**2 - num_tokens), mode='constant')
    attention_map = attn_map.reshape(grid_size, grid_size)

    # Grad-CAM
    wrapped = ViTWrapper(model)
    target_layer = model.vit.encoder.layer[-1]
    gradcam = LayerGradCam(wrapped, target_layer)
    attributions = gradcam.attribute(input_tensor, target=pred_class)

    # Process attributions
    attr = attributions[0].squeeze().detach().cpu().numpy()
    if len(attr.shape) == 2:
        attr = attr.sum(axis=1)
    attr = attr[1:]  # Remove CLS token
    print("Number of patches:", len(attr))

    # Dynamic grid size
    num_patches = len(attr)
    grid_size = int(math.ceil(math.sqrt(num_patches)))
    if num_patches < grid_size**2:
        attr = np.pad(attr, (0, grid_size**2 - num_patches), mode='constant')
    elif num_patches > grid_size**2:
        attr = attr[:grid_size**2]

    # Reshape and normalize
    cam = attr.reshape(grid_size, grid_size)
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

    # Resize to image size
    cam_tensor = torch.tensor(cam).unsqueeze(0).unsqueeze(0)
    cam_resized = transforms.Resize(image.size[::-1])(cam_tensor)[0, 0].numpy()
    cam_resized = np.clip(cam_resized, 0, 1)
    print("cam_resized shape:", cam_resized.shape, "image size:", image.size)
    print("cam_resized min/max:", cam_resized.min(), cam_resized.max())

    # Save raw heatmap for debugging
    heatmap_img = Image.fromarray((cam_resized * 255).astype(np.uint8))
    heatmap_img.save(os.path.join(EXPLAIN_DIR, "raw_heatmap.png"))

    return pred_class, conf, cam_resized, attention_map

# ====================== Overlay Heatmap ======================
def overlay_heatmap(image: Image.Image, heatmap: np.ndarray) -> Image.Image:
    """
    Overlay heatmap on the original image using PIL.

    Args:
        image (Image.Image): Original PIL image.
        heatmap (np.ndarray): Normalized heatmap [0,1], matching image size.

    Returns:
        Image.Image: Blended PIL image.
    """
    # Ensure heatmap is normalized
    heatmap = np.clip(heatmap, 0, 1)
    print("Heatmap min/max after clip:", heatmap.min(), heatmap.max())

    # Create green overlay with alpha based on heatmap
    green = np.zeros((image.size[1], image.size[0], 4), dtype=np.uint8)
    green[:, :, 1] = 255  # Green channel
    green[:, :, 3] = (heatmap * 255 * 0.7).astype(np.uint8)  # Alpha channel (70% max opacity)
    green_img = Image.fromarray(green, mode='RGBA')
    green_img.save(os.path.join(EXPLAIN_DIR, "green_overlay.png"))

    # Convert original image to RGBA
    img_rgba = image.convert("RGBA")

    # Composite images
    blended = Image.alpha_composite(img_rgba, green_img)
    print("Blended image mode:", blended.mode, "size:", blended.size)

    return blended

# ====================== Visualization ======================
def save_explanation(image: Image.Image, cam: np.ndarray, attn_map: np.ndarray, pred_class: int, conf: float, save_path: str):
    # Generate blended image
    blended_img = overlay_heatmap(image, cam)

    # Create figure
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))

    # Plot blended image
    ax[0].imshow(blended_img)
    ax[0].set_title(f"Grad-CAM: {class_names[pred_class]} ({conf:.2f})")
    ax[0].axis("off")

    # Plot attention map
    im = ax[1].imshow(attn_map, cmap='hot', interpolation='bilinear')
    ax[1].set_title("Attention Map")
    ax[1].axis("off")
    plt.colorbar(im, ax=ax[1], fraction=0.046, pad=0.04)

    # Save and close
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

# ====================== Prediction ======================
def predict_image(image: Image.Image):
    pipe = pipeline("image-classification", model=model, feature_extractor=processor, device=0 if torch.cuda.is_available() else -1)
    return pipe(image)

# ====================== API ======================
def scanImage(image: Image.Image):
    pred_class, conf, cam_resized, attn_map = compute_gradcam(image)
    results = predict_image(image)

    explain_path = os.path.join(EXPLAIN_DIR, f"explain_{random.randint(100000,999999)}.png")
    save_explanation(image, cam_resized, attn_map, pred_class, conf, explain_path)

    return {
        "results": results,
        "pred_class": pred_class,
        "confidence": conf,
        "explainability_image": explain_path
    }