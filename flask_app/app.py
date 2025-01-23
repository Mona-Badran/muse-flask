from flask import Flask, request, jsonify
import faiss
import torch
import numpy as np
import pandas as pd
import os
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

app = Flask(__name__)

index_path = "siglip-faiss-wikiart/siglip_10k.index"
csv_path = "siglip-faiss-wikiart/wikiart_final.csv"

index = faiss.read_index(index_path)
metadata = pd.read_csv(csv_path)

processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

@app.route('/')
def home():
    return jsonify({"message": "Flask server is running successfully!"})

# Route to handle image uploads
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    image = request.files['image']
    
    result = {
        'style': 'Impressionism',
        'common_artists': ['Monet', 'Renoir'],
        'description': 'A famous art style from the 19th century.'
    }

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)