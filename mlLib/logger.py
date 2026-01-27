from flask import Flask, request, jsonify
import wandb
import base64
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
import numpy as np

app = Flask(__name__)
wandb.init(project="ml_training_logs")

def calculate_laplacian(img):
    arr = np.array(img, dtype=np.float32)
    
    laplacian_per_channel = []
    for c in range(arr.shape[2]):
        channel = arr[:, :, c]
        laplacian = (
            -4.0 * channel +
            np.roll(channel, 1, axis=1) +  # left
            np.roll(channel, -1, axis=1) +  # right
            np.roll(channel, 1, axis=0) +  # top
            np.roll(channel, -1, axis=0)   # bottom
        )
        laplacian[0, :] = 0
        laplacian[-1, :] = 0
        laplacian[:, 0] = 0
        laplacian[:, -1] = 0
        
        laplacian_per_channel.append(laplacian)
    
    return np.stack(laplacian_per_channel, axis=2)

def visualize_laplacian(laplacian_arr, normalize=True):
    if normalize:
        lap_min = laplacian_arr.min()
        lap_max = laplacian_arr.max()
        if lap_max - lap_min > 0:
            normalized = ((laplacian_arr - lap_min) / (lap_max - lap_min) * 255).astype(np.uint8)
        else:
            normalized = np.zeros_like(laplacian_arr, dtype=np.uint8)
    else:
        normalized = np.clip(laplacian_arr + 127.5, 0, 255).astype(np.uint8)
    
    return Image.fromarray(normalized)

def calculate_laplacian_loss_components(original, prediction):
    lap_orig = calculate_laplacian(original)
    lap_pred = calculate_laplacian(prediction)
    
    diff = np.abs(lap_pred - lap_orig)
    
    total_loss = np.sum(diff)
    
    H, W, C = diff.shape
    valid_pixels = (H - 2) * (W - 2) * C
    avg_loss = total_loss / valid_pixels if valid_pixels > 0 else 0
    
    return lap_orig, lap_pred, diff, total_loss, avg_loss

def create_laplacian_heatmap(diff_arr):
    diff_gray = np.mean(diff_arr, axis=2)
    
    # Normalize to 0-255
    diff_max = diff_gray.max()
    if diff_max > 0:
        normalized = (diff_gray / diff_max * 255).astype(np.uint8)
    else:
        normalized = np.zeros_like(diff_gray, dtype=np.uint8)
    
    heatmap = np.zeros((normalized.shape[0], normalized.shape[1], 3), dtype=np.uint8)
    heatmap[:, :, 0] = normalized  # Red = high difference
    heatmap[:, :, 2] = 255 - normalized  # Blue = low difference
    
    return Image.fromarray(heatmap)

def create_edge_preservation_viz(original, prediction):
    lap_orig = calculate_laplacian(original)
    lap_pred = calculate_laplacian(prediction)
    
    edge_orig = np.sqrt(np.sum(lap_orig ** 2, axis=2))
    edge_pred = np.sqrt(np.sum(lap_pred ** 2, axis=2))
    
    max_edge = max(edge_orig.max(), edge_pred.max())
    if max_edge > 0:
        edge_orig_norm = (edge_orig / max_edge * 255).astype(np.uint8)
        edge_pred_norm = (edge_pred / max_edge * 255).astype(np.uint8)
    else:
        edge_orig_norm = np.zeros_like(edge_orig, dtype=np.uint8)
        edge_pred_norm = np.zeros_like(edge_pred, dtype=np.uint8)
    
    edge_orig_img = np.stack([edge_orig_norm] * 3, axis=2)
    edge_pred_img = np.stack([edge_pred_norm] * 3, axis=2)
    
    return Image.fromarray(edge_orig_img), Image.fromarray(edge_pred_img)

def calculate_frequency_analysis(img):
    arr = np.array(img, dtype=np.float32).mean(axis=2)  # Grayscale
    
    # 2D FFT
    fft = np.fft.fft2(arr)
    fft_shift = np.fft.fftshift(fft)
    magnitude = np.abs(fft_shift)
    
    magnitude_log = np.log1p(magnitude)
    
    mag_normalized = ((magnitude_log - magnitude_log.min()) / 
                      (magnitude_log.max() - magnitude_log.min()) * 255).astype(np.uint8)
    
    freq_viz = np.stack([mag_normalized] * 3, axis=2)
    
    return Image.fromarray(freq_viz)

def create_heatmap(img1, img2):
    arr1 = np.array(img1, dtype=np.float32)
    arr2 = np.array(img2, dtype=np.float32)
    
    diff = np.abs(arr1 - arr2)
    diff_gray = np.mean(diff, axis=2)
    
    diff_normalized = (diff_gray / diff_gray.max() * 255).astype(np.uint8) if diff_gray.max() > 0 else diff_gray.astype(np.uint8)
    
    heatmap = np.zeros((diff_normalized.shape[0], diff_normalized.shape[1], 3), dtype=np.uint8)
    heatmap[:, :, 0] = diff_normalized
    heatmap[:, :, 2] = 255 - diff_normalized
    
    return Image.fromarray(heatmap)

def calculate_metrics(original, prediction):
    arr_orig = np.array(original, dtype=np.float32)
    arr_pred = np.array(prediction, dtype=np.float32)
    
    mae = np.mean(np.abs(arr_orig - arr_pred))
    mse = np.mean((arr_orig - arr_pred) ** 2)
    
    if mse > 0:
        psnr = 20 * np.log10(255.0 / np.sqrt(mse))
    else:
        psnr = float('inf')
    
    return mae, mse, psnr

def add_label(img, text, font_size=32):
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except:
        font = ImageFont.load_default()
    
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    x = (img.width - text_width) // 2
    y = 10
    
    draw.rectangle([x-5, y-5, x+text_width+5, y+text_height+5], fill=(0, 0, 0))
    draw.text((x, y), text, fill=(255, 255, 255), font=font)
    
    return img

def create_difference_visualization(img1, img2, scale=3.0):
    arr1 = np.array(img1, dtype=np.float32)
    arr2 = np.array(img2, dtype=np.float32)
    
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
    laplacian_loss = data.get('laplacian_loss', 0.0)
    ssim_loss = data.get('ssim_loss', 0.0)

    wandb.log({
        "epoch": epoch,
        "loss": loss,
        "learning_rate": learning_rate,
        "time": time,
        "mae_loss": mae_loss,
        "mse_loss": mse_loss,
        "color_loss": color_loss,
        "laplacian_loss": laplacian_loss,
        "ssim_loss": ssim_loss
    })
    return jsonify({"status": "success"}), 200

@app.route('/submitImage', methods=['POST'])
def submit_image():
    input_img_b64 = request.json.get('input_img')
    original_img_b64 = request.json.get('original_img')
    prediction_img_b64 = request.json.get('prediction_img')
    step = request.json.get('step')

    input_rgb = base64.b64decode(input_img_b64)
    original_rgb = base64.b64decode(original_img_b64)
    prediction_rgb = base64.b64decode(prediction_img_b64)
    
    input_img = Image.frombytes('RGB', (800, 600), input_rgb)
    original_img = Image.frombytes('RGB', (800, 600), original_rgb)
    prediction_img = Image.frombytes('RGB', (800, 600), prediction_rgb)

    # Calculate standard metrics
    mae, mse, psnr = calculate_metrics(original_img, prediction_img)

    # Calculate Laplacian loss components
    lap_orig, lap_pred, lap_diff, lap_total_loss, lap_avg_loss = calculate_laplacian_loss_components(
        original_img, prediction_img
    )

    # Create Laplacian visualizations
    lap_orig_viz = visualize_laplacian(lap_orig)
    lap_orig_viz = add_label(lap_orig_viz.copy(), "Ground Truth Laplacian (Edges)")
    
    lap_pred_viz = visualize_laplacian(lap_pred)
    lap_pred_viz = add_label(lap_pred_viz.copy(), "Prediction Laplacian (Edges)")
    
    lap_diff_heatmap = create_laplacian_heatmap(lap_diff)
    lap_diff_heatmap = add_label(lap_diff_heatmap.copy(), "Laplacian Loss Heatmap")

    # Edge preservation visualization
    edge_orig_viz, edge_pred_viz = create_edge_preservation_viz(original_img, prediction_img)
    edge_orig_viz = add_label(edge_orig_viz.copy(), "GT Edge Strength")
    edge_pred_viz = add_label(edge_pred_viz.copy(), "Prediction Edge Strength")

    # Frequency analysis
    freq_orig = calculate_frequency_analysis(original_img)
    freq_orig = add_label(freq_orig.copy(), "GT Frequency Spectrum")
    
    freq_pred = calculate_frequency_analysis(prediction_img)
    freq_pred = add_label(freq_pred.copy(), "Prediction Frequency Spectrum")

    # Original visualizations
    input_vs_pred_heatmap = create_heatmap(input_img, prediction_img)
    input_vs_pred_heatmap = add_label(input_vs_pred_heatmap.copy(), "Input vs Prediction")
    
    original_vs_pred_heatmap = create_heatmap(original_img, prediction_img)
    original_vs_pred_heatmap = add_label(original_vs_pred_heatmap.copy(), "Ground Truth vs Prediction")
    
    input_diff = create_difference_visualization(input_img, prediction_img, scale=5.0)
    input_diff = add_label(input_diff.copy(), "Input - Prediction (5x)")
    
    original_diff = create_difference_visualization(original_img, prediction_img, scale=5.0)
    original_diff = add_label(original_diff.copy(), "GT - Prediction (5x)")

    # Create comprehensive visualization grid (4 columns x 4 rows)
    vis = Image.new('RGB', (800 * 4, 600 * 4))
    
    # Row 1: Original images with labels
    vis.paste(add_label(input_img.copy(), "Noisy Input"), (0, 0))
    vis.paste(add_label(original_img.copy(), "Ground Truth"), (800, 0))
    vis.paste(add_label(prediction_img.copy(), "Denoised Output"), (1600, 0))
    
    # Create metrics panel
    metrics_img = Image.new('RGB', (800, 600), color=(40, 40, 40))
    draw = ImageDraw.Draw(metrics_img)
    try:
        font_large = ImageFont.truetype("arial.ttf", 28)
        font_small = ImageFont.truetype("arial.ttf", 20)
    except:
        font_large = ImageFont.load_default()
        font_small = ImageFont.load_default()
    
    metrics_text = (f"Quality Metrics\n\n"
                   f"MAE: {mae:.2f}\n"
                   f"MSE: {mse:.2f}\n"
                   f"PSNR: {psnr:.2f} dB\n\n"
                   f"Laplacian Loss:\n"
                   f"  Total: {lap_total_loss:.2f}\n"
                   f"  Avg: {lap_avg_loss:.4f}\n\n"
                   f"Step: {step}")
    draw.text((40, 80), metrics_text, fill=(255, 255, 255), font=font_large)
    vis.paste(metrics_img, (2400, 0))
    
    # Row 2: Standard heatmaps and differences
    vis.paste(input_vs_pred_heatmap, (0, 600))
    vis.paste(original_vs_pred_heatmap, (800, 600))
    vis.paste(input_diff, (1600, 600))
    vis.paste(original_diff, (2400, 600))
    
    # Row 3: Laplacian visualizations (edge detection)
    vis.paste(lap_orig_viz, (0, 1200))
    vis.paste(lap_pred_viz, (800, 1200))
    vis.paste(lap_diff_heatmap, (1600, 1200))
    
    # Edge strength comparison
    vis.paste(edge_orig_viz, (2400, 1200))
    
    # Row 4: Frequency analysis and comparison
    vis.paste(freq_orig, (0, 1800))
    vis.paste(freq_pred, (800, 1800))
    vis.paste(edge_pred_viz, (1600, 1800))
    
    # Before/After comparison
    comparison_img = Image.new('RGB', (800, 600))
    comp_arr = np.zeros((600, 800, 3), dtype=np.uint8)
    comp_arr[:, :400] = np.array(input_img)[:, :400]
    comp_arr[:, 400:] = np.array(prediction_img)[:, 400:]
    comp_arr[:, 398:402] = [255, 255, 0]
    comparison_img = Image.fromarray(comp_arr)
    comparison_img = add_label(comparison_img.copy(), "Before / After")
    vis.paste(comparison_img, (2400, 1800))

    # Log everything to Weights & Biases
    wandb.log({
        "step": step,
        "input_image": wandb.Image(input_img, caption="Noisy Input"),
        "original_image": wandb.Image(original_img, caption="Ground Truth"),
        "prediction_image": wandb.Image(prediction_img, caption="Denoised Output"),
        "input_vs_pred_heatmap": wandb.Image(input_vs_pred_heatmap, caption="Input vs Prediction Heatmap"),
        "original_vs_pred_heatmap": wandb.Image(original_vs_pred_heatmap, caption="GT vs Prediction Heatmap"),
        "laplacian_gt": wandb.Image(lap_orig_viz, caption="Ground Truth Laplacian"),
        "laplacian_prediction": wandb.Image(lap_pred_viz, caption="Prediction Laplacian"),
        "laplacian_loss_heatmap": wandb.Image(lap_diff_heatmap, caption="Laplacian Loss Heatmap"),
        "edge_strength_gt": wandb.Image(edge_orig_viz, caption="GT Edge Strength"),
        "edge_strength_pred": wandb.Image(edge_pred_viz, caption="Prediction Edge Strength"),
        "frequency_spectrum_gt": wandb.Image(freq_orig, caption="GT Frequency Spectrum"),
        "frequency_spectrum_pred": wandb.Image(freq_pred, caption="Prediction Frequency Spectrum"),
        "comprehensive_visualization": wandb.Image(vis, caption="Complete Visualization Grid"),
        "before_after_comparison": wandb.Image(comparison_img, caption="Before/After Split"),
        "image_mae": mae,
        "image_mse": mse,
        "image_psnr": psnr,
        "image_laplacian_loss_total": lap_total_loss,
        "image_laplacian_loss_avg": lap_avg_loss
    })
    
    return jsonify({
        "status": "success",
        "metrics": {
            "mae": float(mae),
            "mse": float(mse),
            "psnr": float(psnr),
            "laplacian_loss_total": float(lap_total_loss),
            "laplacian_loss_avg": float(lap_avg_loss)
        }
    }), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)