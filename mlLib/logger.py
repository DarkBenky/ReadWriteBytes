from flask import Flask, request, jsonify
import wandb
import base64
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
import numpy as np

app = Flask(__name__)
wandb.init(project="ml_training_logs")

def create_heatmap(img1, img2):
    """Create a heatmap showing pixel-wise differences between two images."""
    arr1 = np.array(img1, dtype=np.float32)
    arr2 = np.array(img2, dtype=np.float32)
    
    # Calculate absolute difference
    diff = np.abs(arr1 - arr2)
    
    # Average across color channels for grayscale difference
    diff_gray = np.mean(diff, axis=2)
    
    # Normalize to 0-255 range
    diff_normalized = (diff_gray / diff_gray.max() * 255).astype(np.uint8) if diff_gray.max() > 0 else diff_gray.astype(np.uint8)
    
    # Apply colormap (blue = low difference, red = high difference)
    heatmap = np.zeros((diff_normalized.shape[0], diff_normalized.shape[1], 3), dtype=np.uint8)
    heatmap[:, :, 0] = diff_normalized  # Red channel
    heatmap[:, :, 2] = 255 - diff_normalized  # Blue channel (inverted)
    
    return Image.fromarray(heatmap)

def calculate_metrics(original, prediction):
    """Calculate image quality metrics."""
    arr_orig = np.array(original, dtype=np.float32)
    arr_pred = np.array(prediction, dtype=np.float32)
    
    # Mean Absolute Error
    mae = np.mean(np.abs(arr_orig - arr_pred))
    
    # Mean Squared Error
    mse = np.mean((arr_orig - arr_pred) ** 2)
    
    # Peak Signal-to-Noise Ratio
    if mse > 0:
        psnr = 20 * np.log10(255.0 / np.sqrt(mse))
    else:
        psnr = float('inf')
    
    return mae, mse, psnr

def add_label(img, text, font_size=20):
    """Add a text label to the top of an image."""
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except:
        font = ImageFont.load_default()
    
    # Draw text with background
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    x = (img.width - text_width) // 2
    y = 10
    
    draw.rectangle([x-5, y-5, x+text_width+5, y+text_height+5], fill=(0, 0, 0))
    draw.text((x, y), text, fill=(255, 255, 255), font=font)
    
    return img

def create_difference_visualization(img1, img2, scale=3.0):
    """Create an amplified difference visualization."""
    arr1 = np.array(img1, dtype=np.float32)
    arr2 = np.array(img2, dtype=np.float32)
    
    # Calculate difference and amplify
    diff = (arr1 - arr2) * scale + 127.5
    diff = np.clip(diff, 0, 255).astype(np.uint8)
    
    return Image.fromarray(diff)

@app.route('/submitLoss', methods=['POST'])
def submit_loss():
    data = request.json
    epoch = data.get('step')
    loss = data.get('loss')
    learning_rate = data.get('learning_rate')
    time = data.get('time')
    mae_loss = data.get('mae_loss', 0.0)
    mse_loss = data.get('mse_loss', 0.0)
    color_loss = data.get('color_loss', 0.0)

    # Log to Weights & Biases
    wandb.log({
        "epoch": epoch,
        "loss": loss,
        "learning_rate": learning_rate,
        "time": time,
        "mae_loss": mae_loss,
        "mse_loss": mse_loss,
        "color_loss": color_loss
    })
    return jsonify({"status": "success"}), 200

@app.route('/submitImage', methods=['POST'])
def submit_image():
    # 3 images encoded as base64 strings (raw RGB data)
    input_img_b64 = request.json.get('input_img')
    original_img_b64 = request.json.get('original_img')
    prediction_img_b64 = request.json.get('prediction_img')
    step = request.json.get('step')

    # Decode base64 to raw RGB bytes, then create PIL Images
    input_rgb = base64.b64decode(input_img_b64)
    original_rgb = base64.b64decode(original_img_b64)
    prediction_rgb = base64.b64decode(prediction_img_b64)
    
    # Create PIL Images from raw RGB data (800x600x3 bytes)
    input_img = Image.frombytes('RGB', (800, 600), input_rgb)
    original_img = Image.frombytes('RGB', (800, 600), original_rgb)
    prediction_img = Image.frombytes('RGB', (800, 600), prediction_rgb)

    # Calculate quality metrics
    mae, mse, psnr = calculate_metrics(original_img, prediction_img)

    # Create visualizations
    # 1. Heatmap: Input vs Prediction (shows what the denoiser changed)
    input_vs_pred_heatmap = create_heatmap(input_img, prediction_img)
    input_vs_pred_heatmap = add_label(input_vs_pred_heatmap.copy(), "Input vs Prediction")
    
    # 2. Heatmap: Original vs Prediction (shows remaining error)
    original_vs_pred_heatmap = create_heatmap(original_img, prediction_img)
    original_vs_pred_heatmap = add_label(original_vs_pred_heatmap.copy(), "Ground Truth vs Prediction")
    
    # 3. Amplified difference: Input vs Prediction
    input_diff = create_difference_visualization(input_img, prediction_img, scale=5.0)
    input_diff = add_label(input_diff.copy(), "Input - Prediction (5x)")
    
    # 4. Amplified difference: Original vs Prediction
    original_diff = create_difference_visualization(original_img, prediction_img, scale=5.0)
    original_diff = add_label(original_diff.copy(), "GT - Prediction (5x)")
    
    # Create comprehensive visualization grid (3 columns x 3 rows)
    vis = Image.new('RGB', (800 * 3, 600 * 3))
    
    # Row 1: Original images
    vis.paste(add_label(input_img.copy(), "Noisy Input"), (0, 0))
    vis.paste(add_label(original_img.copy(), "Ground Truth"), (800, 0))
    vis.paste(add_label(prediction_img.copy(), "Denoised Output"), (1600, 0))
    
    # Row 2: Heatmaps
    vis.paste(input_vs_pred_heatmap, (0, 600))
    vis.paste(original_vs_pred_heatmap, (800, 600))
    
    # Create metrics panel
    metrics_img = Image.new('RGB', (800, 600), color=(40, 40, 40))
    draw = ImageDraw.Draw(metrics_img)
    try:
        font_large = ImageFont.truetype("arial.ttf", 32)
        font_small = ImageFont.truetype("arial.ttf", 24)
    except:
        font_large = ImageFont.load_default()
        font_small = ImageFont.load_default()
    
    metrics_text = f"Quality Metrics\n\nMAE: {mae:.2f}\nMSE: {mse:.2f}\nPSNR: {psnr:.2f} dB\n\nStep: {step}"
    draw.text((50, 100), metrics_text, fill=(255, 255, 255), font=font_large)
    vis.paste(metrics_img, (1600, 600))
    
    # Row 3: Amplified differences
    vis.paste(input_diff, (0, 1200))
    vis.paste(original_diff, (800, 1200))
    
    # Create comparison: shows denoising effectiveness
    comparison_img = Image.new('RGB', (800, 600))
    comp_arr = np.zeros((600, 800, 3), dtype=np.uint8)
    
    # Left half: input, Right half: prediction
    comp_arr[:, :400] = np.array(input_img)[:, :400]
    comp_arr[:, 400:] = np.array(prediction_img)[:, 400:]
    
    # Add vertical divider
    comp_arr[:, 398:402] = [255, 255, 0]  # Yellow line
    
    comparison_img = Image.fromarray(comp_arr)
    comparison_img = add_label(comparison_img.copy(), "Before / After")
    vis.paste(comparison_img, (1600, 1200))

    # Log everything to Weights & Biases
    wandb.log({
        "step": step,
        "input_image": wandb.Image(input_img, caption="Noisy Input"),
        "original_image": wandb.Image(original_img, caption="Ground Truth"),
        "prediction_image": wandb.Image(prediction_img, caption="Denoised Output"),
        "input_vs_pred_heatmap": wandb.Image(input_vs_pred_heatmap, caption="Input vs Prediction Heatmap"),
        "original_vs_pred_heatmap": wandb.Image(original_vs_pred_heatmap, caption="GT vs Prediction Heatmap"),
        "comprehensive_visualization": wandb.Image(vis, caption="Complete Visualization Grid"),
        "before_after_comparison": wandb.Image(comparison_img, caption="Before/After Split"),
        "image_mae": mae,
        "image_mse": mse,
        "image_psnr": psnr
    })
    
    return jsonify({
        "status": "success",
        "metrics": {
            "mae": float(mae),
            "mse": float(mse),
            "psnr": float(psnr)
        }
    }), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)