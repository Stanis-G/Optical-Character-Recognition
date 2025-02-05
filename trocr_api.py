import base64
import io
import os
from PIL import Image
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from transformers import VisionEncoderDecoderModel, TrOCRProcessor
from utils_inf import inference

load_dotenv()

app = Flask(__name__)

# Load configurations from environment variables
MODEL_NAME = os.getenv("MODEL_NAME", "trocr_model")
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "5000"))

# Load the model and tokenizer
model = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME)
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-small-printed")


@app.route('/generate', methods=['POST'])
def generate():
    data = request.json

    image_base64 = data['image']
    image_bytes = base64.b64decode(image_base64)
    image = Image.open(io.BytesIO(image_bytes))

    _, generated_text = inference(image, model, processor)

    return jsonify({"generated_text": generated_text})

if __name__ == '__main__':
    app.run(host=HOST, port=PORT)
