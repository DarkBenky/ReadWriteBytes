from flask import Flask, request, jsonify
import wandb
import base64
from io import BytesIO
from PIL import Image

app = Flask(__name__)
wandb.init(project="ml_training_logs")

@app.route('/submitLoss', methods=['POST'])
def submit_loss():
    data = request.json
    epoch = data.get('step')
    loss = data.get('loss')
    learning_rate = data.get('learning_rate')
    time = data.get('time')

    # Log to Weights & Biases
    wandb.log({"epoch": epoch, "loss": loss, "learning_rate": learning_rate, "time": time})
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

    # Log images to Weights & Biases
    wandb.log({
        "step": step,
        "input_image": wandb.Image(input_img, caption="Input Image"),
        "original_image": wandb.Image(original_img, caption="Original Image"),
        "prediction_image": wandb.Image(prediction_img, caption="Prediction Image")
    })
    return jsonify({"status": "success"}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)