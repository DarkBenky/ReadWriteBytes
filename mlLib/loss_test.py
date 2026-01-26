import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os

# ------------------ CONFIG ------------------
IMG_WIDTH = 800
IMG_HEIGHT = 600
PATH = "/media/user/2TB Clear/imageData"
FOLDERS = [f.path for f in os.scandir(PATH) if f.is_dir()]

# ------------------ DATA ------------------
def loadRandomImage():
    folder = np.random.choice(FOLDERS)
    high_res = Image.open(os.path.join(folder, "high_res.png")).resize((IMG_WIDTH, IMG_HEIGHT))
    return np.array(high_res, dtype=np.float32) / 255.0

def loadXYImages():
    folder = np.random.choice(FOLDERS)
    high_res = Image.open(os.path.join(folder, "high_res.png")).resize((IMG_WIDTH, IMG_HEIGHT))
    low_res = Image.open(os.path.join(folder, "low_res.png")).resize((IMG_WIDTH, IMG_HEIGHT))
    return (np.array(low_res, dtype=np.float32) / 255.0,
            np.array(high_res, dtype=np.float32) / 255.0)

# ------------------ SIMPLE COLOR LOSS ------------------

def colorLoss(pred, target, return_maps=False):
    """
    Simple parallel-friendly color loss:
    1. Per-pixel chroma difference (prevents gray averaging)
    2. Per-pixel chroma magnitude penalty (forces color presence)
    """
    # Normalize to unit vectors (makes color direction matter more than brightness)
    pred_norm = pred / (np.linalg.norm(pred, axis=2, keepdims=True) + 1e-6)
    target_norm = target / (np.linalg.norm(target, axis=2, keepdims=True) + 1e-6)
    
    # Color direction loss (dot product = 1 when colors match)
    dot = np.sum(pred_norm * target_norm, axis=2)
    direction_map = 1.0 - dot
    direction_loss = np.mean(direction_map)
    
    # Color saturation loss (penalize desaturated predictions)
    pred_sat = np.std(pred, axis=2)  # high when R≠G≠B, low when gray
    target_sat = np.std(target, axis=2)
    saturation_map = np.maximum(0, target_sat - pred_sat)
    saturation_loss = np.mean(saturation_map)
    
    total = direction_loss + (saturation_loss + 1) ** 3
    
    print(f"Direction: {direction_loss:.4f}, Saturation: {saturation_loss:.4f}")
    
    if return_maps:
        return total, direction_map, saturation_map
    return total

# ------------------ VISUALIZATION ------------------

def to_pil(img):
    """Convert float32 [0,1] to PIL Image"""
    return Image.fromarray((np.clip(img, 0.0, 1.0) * 255).astype(np.uint8))

def heatmap(values, colormap='hot'):
    """Convert 2D array to colored heatmap"""
    # Normalize to 0-1
    vmin, vmax = values.min(), values.max()
    if vmax > vmin:
        normalized = (values - vmin) / (vmax - vmin)
    else:
        normalized = np.zeros_like(values)
    
    # Simple hot colormap: black -> red -> yellow -> white
    if colormap == 'hot':
        r = np.clip(normalized * 3, 0, 1)
        g = np.clip(normalized * 3 - 1, 0, 1)
        b = np.clip(normalized * 3 - 2, 0, 1)
        rgb = np.stack([r, g, b], axis=-1)
    elif colormap == 'viridis':
        # Simple blue -> green -> yellow approximation
        r = np.clip(normalized * 2 - 0.5, 0, 1)
        g = np.clip(normalized * 1.5, 0, 1)
        b = np.clip(1 - normalized * 1.5, 0, 1)
        rgb = np.stack([r, g, b], axis=-1)
    
    return to_pil(rgb)

def add_label(img, text, color=(255, 255, 255)):
    """Add text label to image"""
    draw = ImageDraw.Draw(img)
    # Use default font
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
    except:
        font = ImageFont.load_default()
    
    # Draw text with black outline for visibility
    x, y = 10, 10
    draw.text((x-1, y-1), text, font=font, fill=(0, 0, 0))
    draw.text((x+1, y-1), text, font=font, fill=(0, 0, 0))
    draw.text((x-1, y+1), text, font=font, fill=(0, 0, 0))
    draw.text((x+1, y+1), text, font=font, fill=(0, 0, 0))
    draw.text((x, y), text, font=font, fill=color)
    return img

def visualize_loss(target, pred_gray, pred_color):
    """Create comprehensive visualization"""
    
    # Calculate losses with maps
    loss_gray, dir_map_gray, sat_map_gray = colorLoss(pred_gray, target, return_maps=True)
    loss_color, dir_map_color, sat_map_color = colorLoss(pred_color, target, return_maps=True)
    
    # Create 3x3 grid
    w, h = IMG_WIDTH, IMG_HEIGHT
    vis = Image.new("RGB", (w * 3, h * 3), color=(20, 20, 20))
    
    # Row 1: Target, Gray prediction, Color prediction
    img_target = add_label(to_pil(target), "Target", (0, 255, 0))
    img_gray = add_label(to_pil(pred_gray), f"Gray (Loss={loss_gray:.3f})", (255, 100, 100))
    img_color = add_label(to_pil(pred_color), f"Color (Loss={loss_color:.3f})", (100, 255, 100))
    
    vis.paste(img_target, (0, 0))
    vis.paste(img_gray, (w, 0))
    vis.paste(img_color, (w * 2, 0))
    
    # Row 2: Direction loss heatmaps
    dir_heat_gray = add_label(heatmap(dir_map_gray, 'hot'), "Direction Loss (Gray)", (255, 200, 0))
    dir_heat_color = add_label(heatmap(dir_map_color, 'hot'), "Direction Loss (Color)", (255, 200, 0))
    
    vis.paste(Image.new("RGB", (w, h), (40, 40, 40)), (0, h))  # Empty space
    vis.paste(dir_heat_gray, (w, h))
    vis.paste(dir_heat_color, (w * 2, h))
    
    # Row 3: Saturation loss heatmaps
    sat_heat_gray = add_label(heatmap(sat_map_gray, 'viridis'), "Saturation Loss (Gray)", (100, 200, 255))
    sat_heat_color = add_label(heatmap(sat_map_color, 'viridis'), "Saturation Loss (Color)", (100, 200, 255))
    
    vis.paste(Image.new("RGB", (w, h), (40, 40, 40)), (0, h * 2))  # Empty space
    vis.paste(sat_heat_gray, (w, h * 2))
    vis.paste(sat_heat_color, (w * 2, h * 2))
    
    return vis

# ------------------ TEST ------------------
def make_gray(img):
    gray = np.mean(img, axis=2, keepdims=True)
    return np.repeat(gray, 3, axis=2)

def make_noisy_color(img, strength=0.15):
    return np.clip(img + np.random.randn(*img.shape) * strength, 0, 1)

if __name__ == "__main__":
    target = loadRandomImage()
    
    pred_gray = make_gray(target)
    pred_color = make_noisy_color(target)
    
    print("=" * 50)
    print("Gray prediction:")
    loss_gray = colorLoss(pred_gray, target)
    
    print("\nColor prediction:")
    loss_color = colorLoss(pred_color, target)
    
    print("=" * 50)
    print(f"\nFinal Results:")
    print(f"Gray loss:  {loss_gray:.4f}")
    print(f"Color loss: {loss_color:.4f}")
    print(f"Improvement: {((loss_gray - loss_color) / loss_gray * 100):.1f}%")
    
    # Show visualization
    vis = visualize_loss(target, pred_gray, pred_color)
    vis.show()
    
    # Optionally save
    # vis.save("color_loss_visualization.png")